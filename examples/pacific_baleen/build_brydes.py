from build_common import write
R=[]
def r(**k): R.append(k)
OL="https://www.cetus.ucsd.edu/docs/publications/OlesonMMS2003.pdf"
HE="(user PDF) Heimlich et al. 2005 JASA 118(3)"
KE="(user PDF) Kerosky et al. 2012 Deep-Sea Res I"
VG="(user PDF) Viloria-Gomora, Romero-Vivas & Urban 2015 JASA 138(5)"
HB="https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1305505/pdf"
MN="https://www.cetus.ucsd.edu/docs/publications/McDonaldNZ2006.pdf"
# Oleson 2003 (Pacific only)
for ct,n,f,d,ici,h,note in [("Be1",37,"two components: 36.6 [35.7-38.2] and 21.2 [20.0-23.7]","2.7 [1.1-6.7]","1.25 [0.20-4.40] min","no","groups; = swept alternating tonal (Heimlich)"),
 ("Be2",36,"41.7 [40.7-43.3]","1.3 [0.5-3.0]","2.93 [0.35-7.20] min","2-4 harmonics up to 160 Hz","lone whales; = high burst-tonal"),
 ("Be3",18,"25.6 [24.4-26.9]","1.7 [1.0-4.0]","2.28 [0.45-8.65] min","2-3 harmonics","= harmonic tone"),
 ("Be4",7,"60 [59.5-60.2]","1.4 [0.8-3.0]","3.35 [1.03-7.08] min","no",""),
 ("Be5",6,"upswept, 26.2 [26.0-26.8]","1.76 [1.3-2.2]","5.50 [1.50-15.88] min","4-8 harmonics","lone whale"),
 ("Be6",31,"downsweep 208 -> 76","2.83 [0.7-4.9]","0.30 [0.11-0.51] min","no","1-4 sweeps ~0.5 s apart")]:
    r(population="Eastern tropical Pacific",region="ETP (sonobuoys near sighted whales)",call_type=ct,shape="tonal",freq_note=f+" Hz, mean [range]",harmonics=h,duration_s=d,interval_type="inter-call",interval_s=ici,n_calls=n,source="Oleson et al. 2003 Mar Mamm Sci 19(2)",url=OL,verification="full text",notes=note)
for ct,n,f,d,ici in [("Be8a pulsed moan",18,"44.6 [43.0-48.0]; ~13 pulses at 51.6 /s; 5-9 harmonics","0.35 [0.25-0.61]","0.05 [0.03-0.09] min"),("Be8b downsweep",4,"~46","0.37 [0.30-0.41]","0.29 [0.25-0.33] min")]:
    r(population="NW Pacific (Japan)",region="off Japan",call_type=ct,freq_note=f,duration_s=d,interval_type="inter-call",interval_s=ici,n_calls=n,source="Oleson et al. 2003",url=OL,verification="full text")
# Heimlich 2005 (ETP hydrophones, 1999-2001)
H=[("Swept alternating tonal",418,"predominant 36.7+-0.6 (33.5-39.0), 1.1+-0.5 s (100%); secondary sweeps 24.9 -> 18.5 (mean 21.7+-1.1), 1.8+-0.6 s (68%)","2.0+-1.1 (0.3-5.4)","2.9+-1.0 (1.0-21.8) min","46.3% of phrase-type hours; Aug-Feb peak; repetitions in 47%"),
 ("Nonswept alternating tonal",302,"28.6+-0.8 (24.6-32.3) 1.1 s (100%); 15.7+-1.5 (13.5-20.7) 1.4 s (48%, ~0.7 s later); 46.7+-3.3 (37.1-53.9) 1.4 s (41%)","2.0+-0.9 (0.3-5.6)","3.0+-1.9 (1.2-12.6) min","33.1%; repetitions in 76%"),
 ("High burst-tonal",128,"41.7+-1.0 (39.9-46.6) 1.0 s (100%); 31.0+-1.2 0.9 s (58%); 49.4+-1.1 0.5 s (38%)","1.0+-0.5 (0.3-2.8)","3.9+-2.8 (1.4-16.1) s","5.9%; ~Be2; repetition 3.3-4.3 s"),
 ("Low burst-tonal",140,"18.5+-0.8 (16.8-26.7) 1.4 s (100%); 29.8+-2.5 1.0 s (100%); 67.2+-1.6 1.4 s (78%)","2.1+-0.8 (0.4-4.0)","5.8+-4.2 (1.7-33.8) s","6.5%"),
 ("Harmonic tone",114,"fundamental 25.4+-0.6 (20.7-25.8) 2.1 s; harmonics 51.3+-4.4 (24%), 77.3+-1.1 (47%)","2.3+-0.8 (0.5-7.0)","6.3+-6.1 (2.0-44.3) s","8.2%; ~Be3")]
for ct,n,f,d,ici,note in H:
    r(population="Eastern tropical Pacific",region="7 hydrophones 8N-8S, 95-110W",years="1999-2001",call_type=ct,shape="multi-tone phrase",freq_note=f,duration_s=d,interval_type="inter-phrase",interval_s=ici,n_calls=n,between_encounter_variability="phrase types differ by site (CE vs NW etc.) and season",source="Heimlich et al. 2005 JASA",url=HE,verification="full text (user PDF)",notes=note)
r(population="E North Pacific (S California, Baja)",region="Southern California Bight",years="2000-2010",call_type="Be4",shape="upsweep then tone",f_start_hz=55,f_end_hz=57.6,freq_note="kernel: 1 s upsweep 55->57.5, 1 s 57.5->57.6, 1 s tone 57.6 (from 30 calls); harmonics noted",duration_s="~3",source="Kerosky et al. 2012 Deep-Sea Res I",url=KE,verification="full text (user PDF)",notes="seasonal Aug-Dec, increasing presence")
for ct,n,lo,hi,pk,d,comp,note in [("Be4",12,"51+-1","60","57","1.9+-0.2","4-7","21 animals; groups of 2+"),("Be10",7,"89+-10","134+-18","104+-11","0.2+-0.1","2","down then up sweep; components to 500 Hz"),("Be11",2,"114+-3","239+-8","164+-22","0.1+-0.0","1","downsweep 239->114"),("Be12",1,"93","145","108","1.29","1","slight downswept tone 145->93; single call")]:
    r(population="Gulf of California",region="Bahia de La Paz / Gulf of California",call_type=ct,freq_note=f"min {lo}, max {hi}, peak {pk}; components {comp}",duration_s=d,n_calls=n,source="Viloria-Gomora et al. 2015 JASA",url=VG,verification="full text (user PDF)",notes=note,include_v1=("yes" if ct in ("Be4","Be10") else "yes (low n)"))
r(population="Central N Pacific (Kauai)",region="PMRF",years="2011-2022",call_type="Be3-type",shape="impulsive broadband onset then tone/slight downsweep",f_peak_hz="25.6-33",freq_note="10-200 Hz overall",duration_s=1.7,interval_type="inter-call (median / mean)",interval_s="306 / 361 s",n_calls="6042 (150 tracks)",between_encounter_variability="ICI +13.1 s/yr (tentative); 22 calls/h along track",source="Helble et al. 2024 Front Mar Sci 11:1305505",url=HB,verification="full text")
r(population="SW Pacific (New Zealand)",region="Great Barrier Island",years="1997",call_type="Downswept call (Be3-like)",shape="impulsive broadband start then downsweep",f_start_hz=25,f_end_hz=22,interval_type="repeat",interval_s="no regular interval",source="McDonald 2006 NZ J Mar Freshw Res 40:519",url=MN,verification="full text (user PDF)",notes="year-round; McDonald also notes a 22 Hz tonal call (22.1+-0.17 Hz, 4-7 s, repeat 2-5 min) of uncertain species")
write(R,"Brydes","BW","literature_call_parameters_brydes.csv")
