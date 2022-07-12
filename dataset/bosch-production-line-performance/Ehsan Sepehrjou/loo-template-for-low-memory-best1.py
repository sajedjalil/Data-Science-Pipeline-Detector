import re
meminfo = open('/proc/meminfo').read()
matched = re.search(r'^MemTotal:\s+(\d+)', meminfo)
if matched: 
    mem_total_kB = int(matched.groups()[0])