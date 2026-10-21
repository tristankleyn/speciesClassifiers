import csv
cols = ["row_id","class","population","region","sites","years","call_type","unit","shape",
 "f_start_hz","f_start_sd","f_end_hz","f_end_sd","f_peak_hz","f_peak_sd","freq_note","harmonics","pulse_rate_hz",
 "duration_s","duration_sd","interval_type","interval_s","interval_sd","n_calls","n_animals_or_sites",
 "between_encounter_variability","method","source","source_via","url","verification","notes","include_v1"]
def write(R,cls,prefix,fn):
    with open(fn,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for i,row in enumerate(R,1):
            row["row_id"]=f"{prefix}{i:03d}"; row["class"]=cls
            row.setdefault("include_v1","yes"); w.writerow({c:row.get(c,"") for c in cols})
    print(fn,len(R),"rows")
