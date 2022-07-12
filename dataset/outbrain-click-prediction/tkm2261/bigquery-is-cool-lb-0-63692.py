
### STEP1
### Destination Table: outbrain.ad_click

QUERY1 = """
SELECT
    ad_id,
    sum(clicked) AS clicked_sum,
    count(1) AS cnt,
FROM
    [outbrain.clicks_train]
GROUP BY
    ad_id
;
"""

### STEP2
### Destination Table: outbrain.ad_prob

QUERY2 = """
SELECT
  ad_id,
  (clicked_sum + 8 * 0.19364537296143453) / (8 + cnt) AS prob
FROM
  [outbrain.ad_click]
;
"""

### STEP3
### Destination Table: outbrain.clicks_test_prob

QUERY3 = """
SELECT
  c.display_id AS display_id,
  c.ad_id AS ad_id,
  CASE
    WHEN
        s.prob is not null 
    THEN
        s.prob
    ELSE
        0.19364537296143453 END AS prob
FROM
  [outbrain.clicks_test] AS c
LEFT OUTER JOIN 
  [outbrain.ad_prob] AS s
ON
  c.ad_id = s.ad_id
;
"""

### STEP4
### Destination Table: outbrain.clicks_test_prob_txt
QUERY4 = """
SELECT
  display_id,
  GROUP_CONCAT(
    cast(ad_id as String) + ':' + cast(prob as String)
    , ',') AS prob_txt
FROM
  [outbrain.clicks_test_prob]
GROUP BY
  display_id
;
"""

### STEP5
### Destination Table: outbrain.submit_bigquery
UDF5 = """
function sortRank(row, emit) {
  emit({ranking: sortRankHelper(row.prob_txt), display_id: row.display_id});
}

function sortRankHelper(row) {
    var data = String(row).split(',').map(function(ele){
      var tmp = ele.split(':');
      return [tmp[0], parseFloat(tmp[1])];
      });
    data.sort(function(a,b){
      if(a[1] < b[1]) return 1;
      if(a[1] > b[1]) return -1;
      return 0;
    });
    return data.map(function(ele){ return ele[0]; }).join(' ');
}

bigquery.defineFunction(
  'sortRank',
  ['prob_txt', 'display_id'],
  [{name: 'ranking', type: 'string'}, {name: 'display_id', type: 'INTEGER'}],
  sortRank
);

"""

QUERY5 = """
SELECT
  display_id,
  ranking as ad_id
FROM
  sortRank(outbrain.clicks_test_prob_txt)
;
"""