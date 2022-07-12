code = """

using DelimitedFiles
using Statistics

## data wrangling by key

function countryD(data; def=[3;6;])
  x = Dict();
  n = sort(unique(data[:,def[1]])); ##Country
  m = sort(unique(data[:,def[2]])); ##Date
  for iter = n
    x[iter] = Dict();
    d2 = data[data[:,def[1]] .== iter, :];
    for iterb = m
      d3 = d2[d2[:,def[2]] .== iterb, :];
      x[iter][iterb] = d3;
      end;
    end;
  return x, n, m; ##, ll;
  end;

## series matrices

function modelA(a,b,c)
  m = size(c,1);
  n = size(b,1);
  cases = zeros(Int64, n, m);
  fatal = zeros(Int64, n, m);
  for iter1=1:n
    for iter2=1:m 
      cases[iter1, iter2] = a[b[iter1]][c[iter2]][1,9];
      fatal[iter1, iter2] = a[b[iter1]][c[iter2]][2,9];
      end;
    end;
  return cases, fatal;
  end;

## implement Centroids

function dists(d,c)
  n = size(c,1);
  dist = zeros(size(d,1), n);
  for iter = 1:n
##    print(size(d), size(c[1,:]));
    k = d .- c[iter,:]';
    dist[:,iter] = sum(abs.(k), dims=2);
    end;
  return dist;
  end;


function cluster(data, n, i; soft=1.0)
  c = Float64.(data[1:n, :]);
  xd = ones(size(data,1)) .* 1e10;
  for iter=1:i
    d = dists(data, c)
    xd[:] = minimum(d, dims=2);
    for iterc=1:n
      v = d[:,iterc] .<= xd .* soft;
      #xd[v] = d[v,iterc]; ## soft-expanding bug-effect
      c[iterc,:] = mean([data[v,:]; c[iterc,:]'], dims=1);
      end;
      ##xd[:] = minimum(d, dims=2);
    end;
   return c, xd;
   end; 
   
   
## pinball loss
## use pinball.() with vectors
## ? correctness

function pinball(p1, p2, p3, t, w)

  p05 = p1 - t;
  p05 = abs(p05) * (sign(-p05)*0.05 + (p05 >= 0));

  p50 = abs(p2 - t) * 0.5;

  p95 = p3 - t;
  p95 = abs(p95) * (sign(p95)*0.05 + (p95 < 0));

  return ((p05 + p50 + p95) * (w/3));
  end;


## main

d,h = readcsv("train.csv", ',', '\n', header=true);

a,b,c = countryD(copy(d[d:,4].!="US",:]), def=[4;7]); ## separate out US for county-based modelling

print(a["Australia"][c[1]][1,:], " ", c[1]); ## australia day 1 :row 1, day 1 string


group1 = [size(a[x][c[1]],1) for x in b];

b2 = b[group1 .== 2];

caseg1, fatalg1 = modelA(a,b2,c);

xcr, xdr = cluster(caseg1, 20, 50, soft=1.1);

median(xdr)
> 627.5

"""

print(code)
