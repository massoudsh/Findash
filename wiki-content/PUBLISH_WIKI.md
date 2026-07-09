# How to Publish the Wiki to GitHub

The wiki content has been prepared in this folder. Follow these steps to publish it to your GitHub Wiki.

## Step 1: Initialize the Wiki on GitHub

1. Go to your repository: https://github.com/massoudsh/Findash
2. Click on the **Wiki** tab
3. Click **Create the first page**
4. Add a simple title like "Home" and any content (this will be replaced)
5. Click **Save Page**

This creates the wiki repository at `https://github.com/massoudsh/Findash.wiki.git`

## Step 2: Clone and Replace Wiki Content

Run these commands in your terminal:

```bash
# Clone the wiki repository
cd /tmp
git clone https://github.com/massoudsh/Findash.wiki.git
cd Findash.wiki

# Remove the placeholder page
rm -f *.md

# Copy all wiki content
cp /Users/massoudshemirani/MyProjects/Octopus/Findash/wiki-content/*.md .

# Commit and push
git add -A
git commit -m "Add comprehensive wiki documentation"
git push origin main
```

## Step 3: Verify

Visit https://github.com/massoudsh/Findash/wiki to see your wiki!

## Wiki Pages

| Page | Description |
|------|-------------|
| Home.md | Wiki hub and overview |
| Onboarding.md | Step-by-step first-time user onboarding |
| Getting-Started.md | Installation and setup guide |
| Project-Structure.md | Repo layout and quick commands |
| Configuration.md | Environment variables and config |
| Architecture.md | System architecture diagrams |
| AI-Agents.md | Documentation for 11 AI agents |
| Database.md | Database schema and models |
| Data-Sources.md | Market/news providers and API keys |
| API-Reference.md | REST API documentation |
| Frontend.md | Frontend architecture |
| Deployment.md | Production deployment guide |
| Troubleshooting.md | Common issues and fixes |
| Contributing.md | Contribution guidelines |
| _Sidebar.md | Navigation sidebar |
| _Footer.md | Page footer |

## Alternative: One-liner

After creating the first wiki page on GitHub web interface:

```bash
cd /tmp && rm -rf Findash.wiki && git clone https://github.com/massoudsh/Findash.wiki.git && cd Findash.wiki && rm -f *.md && cp /Users/massoudshemirani/MyProjects/Octopus/Findash/wiki-content/*.md . && git add -A && git commit -m "Add comprehensive wiki documentation" && git push origin main && echo "Wiki published successfully!"
```
