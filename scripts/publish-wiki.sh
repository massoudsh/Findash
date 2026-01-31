#!/bin/bash
# Findash Wiki Publishing Script
# Run this after creating the first wiki page on GitHub

set -e

WIKI_CONTENT_DIR="$(dirname "$0")/../wiki-content"
TEMP_DIR="/tmp/findash-wiki-$$"

echo "📚 Findash Wiki Publisher"
echo "========================="
echo ""

# Check if wiki content exists
if [ ! -d "$WIKI_CONTENT_DIR" ]; then
    echo "❌ Wiki content not found at $WIKI_CONTENT_DIR"
    exit 1
fi

echo "✅ Found wiki content"

# Try to clone the wiki repository
echo "📥 Cloning wiki repository..."
if ! git clone https://github.com/massoudsh/Findash.wiki.git "$TEMP_DIR" 2>/dev/null; then
    echo ""
    echo "❌ Wiki repository not found!"
    echo ""
    echo "Please create the first wiki page:"
    echo "1. Go to: https://github.com/massoudsh/Findash/wiki"
    echo "2. Click 'Create the first page'"
    echo "3. Click 'Save Page'"
    echo "4. Run this script again"
    echo ""
    open "https://github.com/massoudsh/Findash/wiki/_new" 2>/dev/null || xdg-open "https://github.com/massoudsh/Findash/wiki/_new" 2>/dev/null
    exit 1
fi

echo "✅ Wiki repository cloned"

# Copy wiki content
echo "📋 Copying wiki content..."
cd "$TEMP_DIR"
rm -f *.md
cp "$WIKI_CONTENT_DIR"/*.md . 2>/dev/null || true
rm -f PUBLISH_WIKI.md  # Remove instructions file if present

# Count files
FILE_COUNT=$(ls -1 *.md 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Copied $FILE_COUNT wiki pages"

# Commit and push
echo "📤 Pushing to GitHub..."
git add -A
git commit -m "Add comprehensive wiki documentation

Pages added:
- Home: Project overview
- Getting Started: Installation guide
- Architecture: System design
- AI Agents: 11 agent documentation
- API Reference: REST API docs
- Database: Schema documentation
- Frontend: UI architecture
- Deployment: Production guide
- Configuration: Environment setup
- Contributing: Contribution guide" 2>/dev/null || echo "No changes to commit"

if git push origin master 2>/dev/null || git push origin main 2>/dev/null; then
    echo ""
    echo "✅ Wiki published successfully!"
    echo ""
    echo "🔗 View your wiki at:"
    echo "   https://github.com/massoudsh/Findash/wiki"
    echo ""
else
    echo "❌ Failed to push. Please check your Git credentials."
    exit 1
fi

# Cleanup
rm -rf "$TEMP_DIR"
