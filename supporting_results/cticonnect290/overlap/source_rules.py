import re
from urllib.parse import urlsplit,urlunsplit,parse_qsl,urlencode

def urlkey(raw):
 try:
  x=urlsplit(raw);host=(x.hostname or '').casefold().removeprefix('www.')
  if x.scheme not in ['http','https'] or not host or x.path in ['','/']:return None
  query=[(k,v) for k,v in parse_qsl(x.query,keep_blank_values=True) if not k.casefold().startswith('utm_') and k.casefold() not in {'fbclid','gclid'}]
  return urlunsplit(('https',host,x.path.rstrip('/'),urlencode(sorted(query)),''))
 except ValueError:return None

def classify(url):
 u=urlsplit(url);host=(u.hostname or '').removeprefix('www.');path=u.path
 patterns=[('specific_advisory_id',r'(?:TALOS-\d{4}-\d+|/GHSA-[\w-]+|ssa-\d+|intel-sa-\d+|BOSCH-SA-\d+|/qsa-\d+-\d+|FG-IR-\d+-\d+|/icsa-\d+-\d+-\d+|AMI-SA-\d+)'),('specific_commit_or_changeset',r'(?:/commit/[0-9a-f]{7,40}|/changeset(?:/|$))'),('specific_disclosure_post',r'(?:/lists/(?:oss-security|fulldisclosure)/\d{4}/|/fulldisclosure/\d{4}/|/files/\d+/|/vuls/id/\d+|/vuls/hitachi-sec-|/security/advisories/GHSA-)')]
 for label,pattern in patterns:
  if re.search(pattern,path,re.I):return True,label
 if host=='cert-in.org.in' and re.search('CIVN-\\d{4}-\\d+',u.query,re.I):return True,'specific_CERT_advisory'
 if host=='ibm.com' and re.search('/support/pages/node/\d+',path):return True,'specific_IBM_bulletin'
 if host=='incibe.es' and '/notices/aviso/' in path:return True,'specific_INCIBE_notice'
 if host=='cert.pl' and re.search('/posts/\d{4}/\d{2}/CVE-',path,re.I):return True,'specific_CERT_case'
 if host=='psirt.bosch.com' and '/security-advisories/' in path:return True,'specific_vendor_advisory'
 if host=='nvidia.custhelp.com' and '/app/answers/detail/a_id/' in path:return True,'specific_vendor_bulletin'
 if host in ['cyberdanube.com','dragos.com','rapid7.com','outpost24.com','sonarsource.com','zyxel.com'] and any(t in path for t in ['/advisory/','/blog/','/blog/post/','/security-advisories/','multiple-vulnerabilities']):return True,'specific_technical_disclosure'
 if host=='qualys.com' and path.endswith('/syslog.txt'):return True,'specific_technical_disclosure'
 if host=='people.redhat.com' and '/~hkario/marvin' in path:return True,'named_cryptographic_attack_disclosure'
 if host=='plugins.trac.wordpress.org' and '/changeset' in path:return True,'specific_changeset'
 return False,'outside_specific_disclosure_rule; generic source/release/product/update URL or unsupported specificity'

