---
layout: page
permalink: /archives/
---

<div>
 <ul class="menu-deroulant">
  {%for post in site.posts %}
      {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
      {% capture nyear %}{{ post.next.date | date: '%Y' }}{% endcapture %}    
      {% capture month %}{{ post.date | date: '%b' }}{% endcapture %}
      {% capture nmonth %}{{ post.next.date | date: '%b' }}{% endcapture %}

      {% if year != nyear %}
        {% if nyear.notblank %}
          </li>
          </ul> <!-- close month -->
          </ul> <!-- close year -->
          </li>
        {% endif %}
          <li> <a class="chevron right" href="#"> {{ post.date | date: '%Y' }}</a> 
          <ul id="{{ post.date | date: '%Y' }}" >
            <li><a class="chevron right" href="#"> {{ post.date | date: '%b' }}</a>
            <ul id="{{ post.date | date: '%Y%b' }}">

      {% else %}
        {% if month != nmonth %}
          </ul>  <!-- close month -->
          </li>
            <li><a class="chevron right" href="#"> {{ post.date | date: '%b' }}</a>
            <ul id="{{ post.date | date: '%Y%b' }}">

        {% endif %}
      {% endif %}

      <li><time>{{ post.date | date:"%d" }}</time>  <a href="{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
  </li>
  </ul>  <!-- close month -->
  </ul>  <!-- close year -->
  </li>
  </ul>  <!-- close menu -->
</div>