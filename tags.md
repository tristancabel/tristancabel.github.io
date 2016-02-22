---
layout: page
permalink: /tags/
---

<div id="archives">
{% for tag in site.tags %}
  <div class="archive-group">
    {% capture tag_name %}{{ tag | first }}{% endcapture %}
    <h3 id="#{{ tag_name | slugize }}">{{ tag_name }}</h3>
    <a name="{{ tag_name | slugize }}"></a>
    <ul>
    {% for post in site.tags[tag_name] %}
    <article class="archive-item">
      <li><h4><a href="{{ post.url  | prepend:site.baseurl}}">{{post.title}}</a></h4></li>
    </article>
    {% endfor %}
    </ul>
 </div>       
{% endfor %}
</div>