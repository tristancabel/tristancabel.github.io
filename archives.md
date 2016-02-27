---
layout: page
permalink: /archives/
---

<div>
 <ul>
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
          <li> <a href="#"> {{ post.date | date: '%Y' }}</a> 
          <ul>
            <li><a href="#"> {{ post.date | date: '%b' }}</a>
            <ul>

      {% else %}
        {% if month != nmonth %}
          </ul>  <!-- close month -->
          </li>
            <li><a href="#"> {{ post.date | date: '%b' }}</a>
            <ul>

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