<nav class="reading-toc" aria-label="分类">
  <div class="reading-toc-title">分类</div>
  <ul>
    <% for (const item of items) { %>
      <% const posts = Array.isArray(item.posts) ? item.posts : []; %>
      <% const slug = item.slug || item.category; %>
      <li>
        <a class="no-external" href="#<%= slug %>">
          <span><%= item.category %></span>
          <span class="reading-toc-count"><%= posts.length %></span>
        </a>
      </li>
    <% } %>
  </ul>
</nav>

::: {.reading-collection}
<% for (const item of items) { %>
<% const posts = Array.isArray(item.posts) ? item.posts : []; %>
<% const slug = item.slug || item.category; %>

::: {.reading-category}
::: {.reading-category-header}
<div class="reading-category-kicker">外部读物</div>

## <%= item.category %> {#<%= slug %>}

<% if (item.description) { %>
<%= item.description %>
<% } %>
:::

<% if (posts.length) { %>
```{=html}
<ol class="reading-link-list">
  <% for (const p of posts) { %>
    <% const tags = Array.isArray(p.tags) ? p.tags : []; %>
    <% const related = Array.isArray(p.related) ? p.related : []; %>
    <li class="reading-link-card">
      <a class="reading-link-title" href="<%= p.href %>"><%= p.title %></a>
      <div class="reading-link-meta">
        <% if (p.site) { %><span><%= p.site %></span><% } %>
        <% if (p.author) { %><span><%= p.author %></span><% } %>
        <% if (p.added) { %><span>收藏于 <%= p.added %></span><% } %>
      </div>
      <% if (p.comment) { %>
        <p class="reading-link-comment"><%= p.comment %></p>
      <% } %>
      <% if (tags.length || related.length) { %>
        <div class="reading-link-tags">
          <% for (const tag of tags) { %><span><%= tag %></span><% } %>
          <% for (const rel of related) { %><a class="no-external" href="/categories/<%= rel %>.html">关联：<%= rel %></a><% } %>
        </div>
      <% } %>
    </li>
  <% } %>
</ol>
```
<% } else { %>
<p class="reading-empty">这个分类先留空，之后读到合适的文章再补。</p>
<% } %>
:::
<% } %>
:::
