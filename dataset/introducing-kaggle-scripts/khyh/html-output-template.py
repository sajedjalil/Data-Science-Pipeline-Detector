'''
- This code is ripped from output of RMarkdown.
https://www.kaggle.io/svf/52473/d9aacb2b33a717f6b51a775ca9e509df/output.html

- This script is under construction.

'''


template_raw = '''

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">

<head>
    {% block head %}
    <meta charset="utf-8">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="generator" content="pandoc" />
    
    <meta name="author" content="{{ author }}" />
    
    <title>{{ title }}</title>
    
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap-theme.min.css">
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
    
    <style type="text/css">code{white-space: pre;}</style>
    
    
    <style type="text/css">
      pre:not([class]) {
        background-color: white;
      }
    </style>
    <script type="text/javascript">
    if (window.hljs && document.readyState && document.readyState === "complete") {
       window.setTimeout(function() {
          hljs.initHighlighting();
       }, 0);
    }
    </script>
    {% endblock %}
</head>


<body>

<style type = "text/css">
.main-container {
  max-width: 940px;
  margin-left: auto;
  margin-right: auto;
}
code {
  color: inherit;
  background-color: rgba(0, 0, 0, 0.04);
}
img { 
  max-width:100%; 
  height: auto; 
}
</style>
<div class="container-fluid main-container">

<div id="header">
{% block intro %}
    <h1 class="title">{{ title }}</h1>
    <h4 class="author"><em>{{ author }}</em></h4>
    <h4 class="date"><em>{{ date }}</em></h4>
    <p>description description description description description description</p>
    <p>description description description description description description</p>
    <p>description description description description description description</p>
{% endblock %}
</div>

<!-- repeat this -->

<div id="{{ sub_title }}" class="section level3">
<h3>{{ sub_title }}</h3>

<p>description description description description description description</p>
<p>description description description description description description</p>
<p>description description description description description description</p>
</div>

<!-- /repeat this -->


<script>

// add bootstrap table styles to pandoc tables
$(document).ready(function () {
  $('tr.header').parent('thead').parent('table').addClass('table table-condensed');
});

</script>

<!-- dynamically load mathjax for compatibility with self-contained -->
<script>
  (function () {
    var script = document.createElement("script");
    script.type = "text/javascript";
    script.src  = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML";
    document.getElementsByTagName("head")[0].appendChild(script);
  })();
</script>

</body>
</html>
'''

from jinja2 import Template
import time

template = Template(template_raw)

with open("results.html","wb") as outfile:
    outfile.write(template.render(
        title='title of this script',
        author='your name here',
        date=time.asctime(),
        sub_title='sub title'
        ).encode("utf-8"))