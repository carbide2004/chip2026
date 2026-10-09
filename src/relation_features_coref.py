"""Seeded patient-coreference features built from official patient mentions."""
from __future__ import annotations
import math
from typing import Any,Dict,List,Sequence,Tuple
from .relation_features import association_set,pair_features,positive_occurrences,passage_index

def coref_feature_names()->List[str]:
 return ['local_target_any','local_target_fraction','local_other_any','local_other_fraction','local_target_only_fraction','same_passage_target_any','same_passage_target_fraction','same_passage_target_only_fraction','previous_anchor_target_fraction','next_anchor_target_fraction','nearest_anchor_target_fraction','target_anchor_precedes_fraction','mean_passage_gap_log']

def coref_features(doc:Dict[str,Any],patient:Dict[str,Any],occurrences:Sequence[Dict[str,Any]],radius:int=500)->List[float]:
 target=str(patient['patient_id']);anchors=[]
 for p in doc.get('patient',[]):
  owner=str(p['patient_id'])
  for mention in p.get('mention',[]):anchors.append((int(mention['offset']),owner,passage_index(doc,int(mention['offset']))))
 anchors.sort();count=max(1,len(occurrences));local_t=[];local_o=[];local_only=[];same_t=[];same_only=[];previous=[];following=[];nearest=[];precedes=[];passage_gaps=[]
 for occurrence in occurrences:
  offset=int(occurrence['offset']);pidx=occurrence['passage_index'];local=[a for a in anchors if abs(a[0]-offset)<=radius];owners={a[1] for a in local};local_t.append(target in owners);local_o.append(any(x!=target for x in owners));local_only.append(owners=={target});same=[a for a in anchors if a[2]==pidx];same_owners={a[1] for a in same};same_t.append(target in same_owners);same_only.append(same_owners=={target});before=[a for a in anchors if a[0]<=offset];after=[a for a in anchors if a[0]>offset];previous.append(bool(before) and before[-1][1]==target);following.append(bool(after) and after[0][1]==target)
  if anchors:
   closest=min(anchors,key=lambda a:abs(a[0]-offset));nearest.append(closest[1]==target);precedes.append(closest[1]==target and closest[0]<=offset);target_passages=[a[2] for a in anchors if a[1]==target and a[2] is not None];passage_gaps.append(min((abs(pidx-x) for x in target_passages),default=1000) if pidx is not None else 1000)
  else:nearest.append(False);precedes.append(False);passage_gaps.append(1000)
 return [float(any(local_t)),sum(local_t)/count,float(any(local_o)),sum(local_o)/count,sum(local_only)/count,float(any(same_t)),sum(same_t)/count,sum(same_only)/count,sum(previous)/count,sum(following)/count,sum(nearest)/count,sum(precedes)/count,math.log1p(sum(passage_gaps)/count)]

def build_examples_coref(docs:Sequence[Dict[str,Any]])->Tuple[List[List[float]],List[int],List[Tuple[str,str,str]]]:
 features=[];labels=[];keys=[]
 for doc in docs:
  occurrences=positive_occurrences(doc);gold={str(row['patient_id']):association_set(row) for row in doc.get('association',[])}
  for patient in doc.get('patient',[]):
   patient_id=str(patient['patient_id'])
   for value,evidence in occurrences.items():
    features.append(pair_features(doc,patient,evidence)+coref_features(doc,patient,evidence));labels.append(int(value in gold.get(patient_id,set())));keys.append((str(doc['pmc_id']),patient_id,value))
 return features,labels,keys
