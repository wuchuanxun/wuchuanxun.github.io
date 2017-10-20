---
layout: page
title: About
description: 志同着，不以千里为远
keywords: Chuanxun Wu
menu: 关于
permalink: /about/
---

我是学习小天使，学习使我快乐 

## 联系

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
  {% endfor %}
* Contact me:  <wuchuanxun@outlook.com>

## Skill Keywords

{% for category in site.data.skills %}
### {{ category.name }}
<div class="btn-inline">
{% for keyword in category.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}
