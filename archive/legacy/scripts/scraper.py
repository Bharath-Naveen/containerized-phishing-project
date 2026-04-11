from urllib.parse import urlparse

def detect_hosting_platform(url: str):
    host = urlparse(url).netloc.lower()

    platforms = {
        "github.io": ("github_pages", "static_site_host"),
        "vercel.app": ("vercel", "static_site_host"),
        "netlify.app": ("netlify", "static_site_host"),
        "pages.dev": ("cloudflare_pages", "static_site_host"),
        "firebaseapp.com": ("firebase", "cloud_hosting"),
        "web.app": ("firebase", "cloud_hosting"),
        "framer.app": ("framer", "design_builder"),
        "carrd.co": ("carrd", "design_builder"),
        "notion.site": ("notion", "design_builder"),
        "replit.app": ("replit", "dev_platform"),
        "glitch.me": ("glitch", "dev_platform"),
        "codesandbox.io": ("codesandbox", "dev_platform"),
    }

    for domain, info in platforms.items():
        if host.endswith(domain):
            return info

    return ("unknown", "unknown")