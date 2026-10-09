"""Conservative expansion of patient mention seeds from repeated aliases in a document."""
from __future__ import annotations
from copy import deepcopy
import re
from typing import Any,Dict
ALIAS_RE=re.compile(r"\b(?:the\s+)?(?:patient|case|proband|individual|subject)\s+(?:[A-Z]?[IVX]+[-.:]?\d+|[A-Z]?\d+|one|two|three|four)\b|\b(?:[IVX]+[-:]\d+|P\d+)\b|\b(?:the\s+)?(?:patient|proband|mother|father|brother|sister|son|daughter)\b",re.I)
WORDS={'one':'1','two':'2','three':'3','four':'4'}
def norm(text:str)->str:return re.sub(r'\s+',' ',text.casefold().strip())
def key(text:str):
 s=norm(text);s=re.sub(r'^the\s+','',s)
 m=re.fullmatch(r'(?:patient|case|individual|subject|proband|p)\s*([0-9]+|one|two|three|four)',s)
 if m:return ('number',WORDS.get(m.group(1),m.group(1)))
 m=re.fullmatch(r'([ivx]+)[-:](\d+)',s)
 if m:return ('pedigree',m.group(1),m.group(2))
 if s in {'mother','father','brother','sister','son','daughter'}:return ('role',s)
 if s=='proband':return ('proband',)
 if s=='patient':return ('patient',)
 return ('exact',s)
def expand_patient_mentions(doc:Dict[str,Any])->Dict[str,Any]:
 output=deepcopy(doc);owners={};patient_count=len(output.get('patient',[]))
 for patient in output.get('patient',[]):
  pid=str(patient['patient_id'])
  for mention in patient.get('mention',[]):owners.setdefault(key(mention['text']),set()).add(pid)
 unique={k:next(iter(v)) for k,v in owners.items() if len(v)==1};by_id={str(p['patient_id']):p for p in output.get('patient',[])};existing={(int(m['offset']),int(m.get('length',len(m['text'])))) for p in output.get('patient',[]) for m in p.get('mention',[])}
 for passage in output.get('full_text',[]):
  for match in ALIAS_RE.finditer(passage.get('text','')):
   alias_key=key(match.group());pid=None if (patient_count>1 and alias_key==('patient',)) else unique.get(alias_key)
   if pid is None and patient_count==1 and alias_key in {('patient',),('proband',)}:pid=str(output['patient'][0]['patient_id'])
   if pid is None:continue
   offset=int(passage['offset'])+match.start();length=match.end()-match.start()
   if (offset,length) in existing:continue
   by_id[pid].setdefault('mention',[]).append({'offset':offset,'length':length,'text':match.group()});existing.add((offset,length))
 for patient in output.get('patient',[]):patient['mention'].sort(key=lambda x:(x['offset'],x.get('length',len(x['text']))))
 return output
