# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "feedparser",
#     "openai",
# ]
# ///
"""Daily newsletter: fetch RSS, summarize with DeepSeek, email via QQ Mail."""

import json
import logging
import os
import re
import smtplib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import feedparser
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "daily_newsletter.log"
SENT_URLS_FILE = SCRIPT_DIR / "sent_urls.json"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("newsletter")

# Load .env file if present (so scheduled tasks can find secrets)
def _load_dotenv(path: Path) -> None:
    """Parse KEY=VALUE pairs from a .env file into os.environ (simple, no deps)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:  # never override existing env vars
            os.environ[key] = val

_load_dotenv(SCRIPT_DIR / ".env")

# API / SMTP secrets from env
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
SMTP_USER = os.environ.get("SMTP_USER", os.environ.get("GMAIL_USER", ""))
SMTP_PASS = os.environ.get("SMTP_PASS", os.environ.get("GMAIL_APP_PASSWORD", ""))
TO_EMAIL = os.environ.get("TO_EMAIL", "2326405860@qq.com")

# RSS feeds and their categories
FEEDS = [
    ("Hacker News", "https://hnrss.org/frontpage?count=10", "Tech"),
    ("MIT Tech Review", "https://www.technologyreview.com/feed/", "AI"),
    ("OpenAI Blog", "https://openai.com/blog/rss.xml", "AI"),
    ("少数派", "https://sspai.com/feed", "Tech"),
    ("IT之家", "https://www.ithome.com/rss/", "Tech"),
    ("36氪", "https://36kr.com/feed", "Finance"),
    ("华尔街见闻", "https://feeds.feedburner.com/wallstreetcn", "Finance"),
    ("第一财经", "https://www.yicai.com/feed/", "Finance"),
]

MAX_ARTICLES_PER_FEED = 8
SUMMARIZE_MODEL = "deepseek-chat"


# ── RSS fetching ─────────────────────────────────────────────────────────────
def fetch_feed(name: str, url: str, category: str) -> list[dict]:
    """Fetch one feed, return list of article dicts. Returns [] on failure."""
    try:
        parsed = feedparser.parse(url)
        if parsed.bozo and parsed.bozo_exception:
            log.warning("Feed parse warning [%s]: %s", name, parsed.bozo_exception)
        articles = []
        for entry in parsed.entries[:MAX_ARTICLES_PER_FEED]:
            link = entry.get("link", "")
            if not link:
                continue
            articles.append({
                "title": entry.get("title", "Untitled").strip(),
                "link": link.strip(),
                "source": name,
                "category": category,
                "published": entry.get("published", ""),
            })
        log.info("[OK] %s: %d articles fetched", name, len(articles))
        return articles
    except Exception as e:
        log.error("[SKIP] %s: %s", name, e)
        return []


def fetch_all_feeds() -> list[dict]:
    """Fetch all feeds in parallel, return deduplicated article list."""
    all_articles = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_feed, name, url, cat): name for name, url, cat in FEEDS}
        for fut in as_completed(futures):
            all_articles.extend(fut.result())

    # Deduplicate by link within this run
    seen = set()
    unique = []
    for a in all_articles:
        if a["link"] not in seen:
            seen.add(a["link"])
            unique.append(a)
    return unique


# ── Dedup across days ────────────────────────────────────────────────────────
def load_sent_urls() -> set[str]:
    """Load previously sent URLs."""
    if SENT_URLS_FILE.exists():
        try:
            data = json.loads(SENT_URLS_FILE.read_text(encoding="utf-8"))
            return set(data.get("urls", []))
        except Exception:
            return set()
    return set()


def save_sent_urls(urls: set[str]) -> None:
    """Save sent URLs (keep last 5000 to bound file size)."""
    trimmed = list(urls)[-5000:]
    SENT_URLS_FILE.write_text(
        json.dumps({"urls": trimmed, "updated": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Summarization ────────────────────────────────────────────────────────────
SUMMARIZE_SYSTEM = """你是一个专业的科技/金融新闻编辑。请用中文总结以下新闻，要求：
- 40-60个词
- 客观、准确
- 突出核心信息
- 只返回摘要文本，不要加"摘要："等前缀
- 如果无法确定内容，返回"无法获取内容"
"""

def summarize_article(client: OpenAI, article: dict) -> str | None:
    """Summarize one article. Returns summary string or None on failure."""
    try:
        resp = client.chat.completions.create(
            model=SUMMARIZE_MODEL,
            max_tokens=200,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": f"标题：{article['title']}\n来源：{article['source']}"},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error("[SKIP summary] %s: %s", article["link"][:80], e)
        return None


def summarize_all(articles: list[dict], sent_urls: set[str]) -> list[dict]:
    """Summarize new articles, skip already-sent ones."""
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    new_articles = [a for a in articles if a["link"] not in sent_urls]
    log.info("New articles to summarize: %d (filtered %d already sent)",
             len(new_articles), len(articles) - len(new_articles))

    results = []
    for i, a in enumerate(new_articles):
        summary = summarize_article(client, a)
        if summary and summary != "无法获取内容":
            a["summary"] = summary
            results.append(a)
    return results


# ── Email composition & sending ──────────────────────────────────────────────
def build_email(articles: list[dict]) -> str:
    """Build HTML email body, grouped by category."""
    today = datetime.now().strftime("%Y年%m月%d日")
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][datetime.now().weekday()]

    cat_config = {
        "AI":  {"title": "AI 人工智能", "icon": "🤖", "color": "#7c3aed"},
        "Tech": {"title": "科技", "icon": "💻", "color": "#2563eb"},
        "Finance": {"title": "金融", "icon": "💰", "color": "#059669"},
    }

    grouped: dict[str, list[dict]] = {"AI": [], "Tech": [], "Finance": []}
    for a in articles:
        cat = a.get("category", "Tech")
        if cat not in grouped:
            cat = "Tech"
        grouped[cat].append(a)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f5;padding:20px 0;">
<tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);">

  <!-- Header -->
  <tr><td style="background:linear-gradient(135deg,#1e1b4b,#312e81);padding:32px 40px;text-align:center;">
    <div style="font-size:13px;color:#a5b4fc;letter-spacing:2px;margin-bottom:6px;">DAILY BRIEFING</div>
    <div style="font-size:22px;font-weight:700;color:#fff;line-height:1.4;">每日 AI · 科技 · 金融速报</div>
    <div style="font-size:14px;color:#c7d2fe;margin-top:8px;">{today} {weekday} · {len(articles)} 条精选</div>
  </td></tr>

  <!-- Body -->
  <tr><td style="padding:24px 32px 8px;">
"""

    for cat_key in ["AI", "Tech", "Finance"]:
        items = grouped.get(cat_key, [])
        if not items:
            continue
        cfg = cat_config[cat_key]

        html += f"""
    <!-- {cfg['title']} -->
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;">
      <tr><td style="padding-bottom:12px;border-bottom:2px solid {cfg['color']};">
        <span style="font-size:20px;">{cfg['icon']}</span>
        <span style="font-size:17px;font-weight:700;color:#1e1b4b;vertical-align:middle;">{cfg['title']}</span>
        <span style="font-size:12px;color:#9ca3af;margin-left:8px;">{len(items)} 篇</span>
      </td></tr>
"""

        for i, a in enumerate(items):
            num = f"{i + 1:02d}"
            title = a['title'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            summary = a['summary'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            link = a['link'].replace('&', '&amp;')
            source = a['source']

            html += f"""
      <tr><td style="padding:14px 0;border-bottom:1px solid #f0f0f0;">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td style="width:28px;vertical-align:top;padding-top:2px;">
              <span style="font-size:12px;font-weight:700;color:{cfg['color']};background:{cfg['color']}10;padding:2px 6px;border-radius:4px;">{num}</span>
            </td>
            <td style="vertical-align:top;">
              <a href="{link}" style="font-size:15px;font-weight:600;color:#111827;text-decoration:none;line-height:1.4;">{title}</a>
              <span style="font-size:11px;color:#9ca3af;margin-left:6px;white-space:nowrap;">— {source}</span>
              <div style="font-size:13px;color:#4b5563;line-height:1.65;margin-top:4px;">{summary}</div>
            </td>
          </tr>
        </table>
      </td></tr>
"""
        html += "</table>\n"

    html += f"""
  </td></tr>

  <!-- Footer -->
  <tr><td style="background:#fafafa;padding:20px 32px;text-align:center;border-top:1px solid #eee;">
    <div style="font-size:11px;color:#9ca3af;">
      由 Claude AI 自动生成 · 每日 8:00 发送<br>
      数据来源：Hacker News · MIT Tech Review · OpenAI · Reuters · Bloomberg
    </div>
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""

    return html


def send_email(body: str, subject: str | None = None) -> bool:
    """Send email via QQ Mail SMTP."""
    if not SMTP_USER or not SMTP_PASS:
        log.error("SMTP_USER / SMTP_PASS not set. Cannot send email.")
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    subject = subject or f"每日速报 — {today}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = TO_EMAIL
    # Plain text fallback first (goes before HTML for MIME alternative)
    plain = re.sub(r"<style[^<]*</style>", "", body, flags=re.DOTALL)
    plain = re.sub(r"<[^>]+>", "", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain).strip()
    msg.attach(MIMEText(plain, "plain", "utf-8"))

    # HTML version last (preferred by clients)
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.qq.com", 587, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        log.info("[SENT] Email delivered to %s", TO_EMAIL)
        return True
    except Exception as e:
        log.error("[EMAIL FAIL] %s", e)
        return False


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    log.info("=== Daily newsletter start ===")

    if not DEEPSEEK_KEY:
        log.error("DEEPSEEK_API_KEY not set. Abort.")
        return 1

    # 1. Fetch RSS
    all_articles = fetch_all_feeds()
    if not all_articles:
        log.warning("No articles fetched from any feed. Exiting.")
        return 0
    log.info("Total articles fetched (deduped): %d", len(all_articles))

    # 2. Dedup against sent history
    sent_urls = load_sent_urls()

    # 3. Summarize new ones
    summarized = summarize_all(all_articles, sent_urls)
    if not summarized:
        log.info("No new articles to send. Done.")
        return 0
    log.info("Summarized %d articles", len(summarized))

    # 4. Build & send email
    email_body = build_email(summarized)
    success = send_email(email_body)

    # 5. Update sent cache
    if success:
        new_sent = sent_urls | {a["link"] for a in summarized}
        save_sent_urls(new_sent)
        log.info("Sent URLs cache updated. Total: %d", len(new_sent))
    else:
        log.warning("Email send failed — sent URLs cache NOT updated for retry next run")

    log.info("=== Daily newsletter end ===")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
