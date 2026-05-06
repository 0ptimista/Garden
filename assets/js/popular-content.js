(() => {
  const ready = (callback) => {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback, { once: true });
      return;
    }

    callback();
  };

  const formatViews = (value) => {
    const number = Number(value);

    if (!Number.isFinite(number) || number <= 0) {
      return "";
    }

    return new Intl.NumberFormat("zh-Hans", {
      notation: number >= 10000 ? "compact" : "standard",
      maximumFractionDigits: 1,
    }).format(number);
  };

  const createItem = (item, index) => {
    const li = document.createElement("li");
    li.className = "home-popular-item";

    const rank = document.createElement("span");
    rank.className = "home-popular-rank";
    rank.textContent = String(index + 1).padStart(2, "0");

    const link = document.createElement("a");
    link.className = "home-popular-link";
    link.href = new URL(item.url, window.location.origin).toString();
    link.textContent = item.title;

    const views = formatViews(item.views);
    const count = document.createElement("span");
    count.className = "home-popular-count";
    count.textContent = views ? `${views} 次` : "";

    li.append(rank, link, count);
    return li;
  };

  const fetchJson = async (url) => {
    const response = await fetch(url, {
      credentials: "omit",
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`Failed to load ${url}: ${response.status}`);
    }

    return response.json();
  };

  const loadPopularPosts = async (container) => {
    const primary = container.dataset.popularSrc || "/popular.json";
    const fallback = container.dataset.popularFallbackSrc || "/assets/data/popular-posts.json";

    try {
      return await fetchJson(primary);
    } catch (error) {
      console.info(error.message);
      return fetchJson(fallback);
    }
  };

  const hidePopularSection = (container) => {
    const section = container.closest("section");
    (section || container).hidden = true;
  };

  ready(async () => {
    const container = document.querySelector("[data-popular-posts]");

    if (!container) {
      return;
    }

    const list = container.querySelector(".home-popular-list");

    if (!list) {
      return;
    }

    try {
      const data = await loadPopularPosts(container);
      const limit = Number(container.dataset.popularLimit || 8);
      const items = Array.isArray(data.items) ? data.items.slice(0, limit) : [];

      if (!items.length) {
        hidePopularSection(container);
        return;
      }

      const fragment = document.createDocumentFragment();
      items.forEach((item, index) => fragment.append(createItem(item, index)));
      list.replaceChildren(fragment);
      container.dataset.ready = "true";

      const updated = container.querySelector("[data-popular-updated]");
      if (updated && data.updatedAt) {
        updated.dateTime = data.updatedAt;
        updated.textContent = new Date(data.updatedAt).toLocaleDateString("zh-Hans", {
          month: "short",
          day: "numeric",
        });
        const meta = updated.closest(".home-popular-meta");
        if (meta) {
          meta.hidden = false;
        }
      }
    } catch (error) {
      console.info(error.message);
      hidePopularSection(container);
    }
  });
})();
