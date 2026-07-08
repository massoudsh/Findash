#!/bin/bash
# =========================================================
# TASK-024 — پشتیبان‌گیری خودکار دیتابیس
# اجرا: هر شب ساعت ۳ بامداد (crontab: 0 3 * * *)
# تنظیم cron: crontab -e  →  0 3 * * * /opt/findash/scripts/backup-db.sh
# =========================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────
BACKUP_DIR="${BACKUP_DIR:-/opt/backups/findash}"
DB_CONTAINER="${DB_CONTAINER:-octopus-db}"
DB_NAME="${DB_NAME:-trading_db}"
DB_USER="${DB_USER:-postgres}"
KEEP_DAYS="${KEEP_DAYS:-30}"

# آروان‌کلاد Object Storage (اختیاری)
ARVAN_BUCKET="${ARVAN_BUCKET:-}"
ARVAN_ENDPOINT="${ARVAN_ENDPOINT:-https://s3.ir-thr-at1.arvanstorage.ir}"
ARVAN_ACCESS_KEY="${ARVAN_ACCESS_KEY:-}"
ARVAN_SECRET_KEY="${ARVAN_SECRET_KEY:-}"

LOG_FILE="${BACKUP_DIR}/backup.log"
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/findash-${DATE}.sql.gz"

# ── Init ──────────────────────────────────────────────────
mkdir -p "$BACKUP_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "━━━━ شروع پشتیبان‌گیری ━━━━"
log "دیتابیس: $DB_NAME | container: $DB_CONTAINER"

# ── Dump ──────────────────────────────────────────────────
if ! docker exec "$DB_CONTAINER" pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_FILE"; then
    log "❌ خطا در pg_dump"
    exit 1
fi

BACKUP_SIZE=$(du -sh "$BACKUP_FILE" | cut -f1)
log "✅ backup ساخته شد: $BACKUP_FILE ($BACKUP_SIZE)"

# ── Upload to Arvan (optional) ────────────────────────────
if [[ -n "$ARVAN_BUCKET" && -n "$ARVAN_ACCESS_KEY" ]]; then
    if command -v aws &>/dev/null; then
        log "آپلود به آروان‌کلاد: $ARVAN_BUCKET"
        AWS_ACCESS_KEY_ID="$ARVAN_ACCESS_KEY" \
        AWS_SECRET_ACCESS_KEY="$ARVAN_SECRET_KEY" \
        aws s3 cp "$BACKUP_FILE" \
            "s3://${ARVAN_BUCKET}/db-backups/$(basename $BACKUP_FILE)" \
            --endpoint-url "$ARVAN_ENDPOINT" \
            --no-verify-ssl 2>/dev/null && \
            log "✅ آپلود موفق" || \
            log "⚠️  آپلود ناموفق — فایل محلی حفظ شده"
    else
        log "⚠️  aws cli نصب نیست — آپلود رد شد"
    fi
fi

# ── Cleanup old backups ───────────────────────────────────
DELETED=$(find "$BACKUP_DIR" -name "findash-*.sql.gz" -mtime +"$KEEP_DAYS" -print -delete | wc -l)
[[ "$DELETED" -gt 0 ]] && log "🗑️  $DELETED فایل قدیمی حذف شد (بیش از $KEEP_DAYS روز)"

# ── Summary ───────────────────────────────────────────────
TOTAL=$(find "$BACKUP_DIR" -name "findash-*.sql.gz" | wc -l)
log "📦 مجموع backup های موجود: $TOTAL"
log "━━━━ پشتیبان‌گیری کامل شد ━━━━"
