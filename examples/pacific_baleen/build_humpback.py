from build_common import write
R=[]
def r(**k): R.append(k)
F15="(user PDF) Fournet, Szabo & Mellinger 2015 JASA 137(1)"
# Fournet 2015 Table IV: type, n, peak, peak_sd, dur, dur_sd, band, band_sd, start, start_sd, class
T=[("Growl",100,128,75,1.0,0.7,257,266,90,39,"LFH trilled"),("Whup",33,132,81,0.7,0.2,249,82,112,158,"LFH trilled"),
("Modulated moan",4,223,102,1.3,0.6,770,173,170,33,"LFH complex"),("Descending moan",4,147,28,1.4,0.6,553,520,225,91,"LFH complex"),
("Groan",3,209,None,3.3,None,434,None,207,None,"LFH simple (SD row misaligned in source)"),("Variable moan",3,180,None,2.5,None,477,None,341,None,"LFH simple (SD row misaligned)"),
("Swop",57,328,287,0.6,0.3,894,717,244,242,"pulsed complex"),("Teepee",12,191,101,0.6,None,327,253,142,None,"pulsed complex (duration SD garbled)"),
("Horse",4,1732,3046,2.1,0.6,3047,3337,882,1115,"pulsed complex"),("Droplet",9,309,140,0.5,0.1,763,795,238,146,"pulsed simple"),
("Squeegie",8,766,698,2.2,1,1177,1177,1006,950,"noisy/complex variable"),("Trumpet",5,859,491,0.9,0.4,1126,604,724,200,"noisy/complex variable"),
("Ahooga",5,380,134,1.1,0.5,480,647,269,68,"noisy/complex variable"),("Ascending shriek",5,1701,1279,1.6,0.7,6781,5204,1459,974,"NC harmonic"),
("Descending shriek",4,1506,1452,0.9,0.2,2740,2419,1302,1197,"NC harmonic"),("Feeding call",43,460,119,16.6,23.1,101,53,459,106,"tonal (multi-unit feeding call)")]
for ct,n,pk,pksd,d,dsd,bw,bwsd,st,stsd,cl in T:
    r(population="N Pacific - SE Alaska (feeding)",region="Frederick Sound / Glacier Bay area, SE Alaska",years="Jun-Sep 2012",call_type=ct,shape=cl,f_start_hz=st,f_start_sd=stsd,f_peak_hz=pk,f_peak_sd=pksd,freq_note=f"bandwidth {bw} (SD {bwsd})",duration_s=d,duration_sd=dsd,n_calls=n,source="Fournet, Szabo & Mellinger 2015 JASA",url=F15,verification="full text (user PDF)",notes="16 call types; overall start F0 277+-398 Hz, duration median 1.1 s")
for ct,n,lo,pk,d,lo2,pk2,d2 in [("Growl",78,"35.8+-21.8","116+-62.6","0.7+-0.3",0,0,0),("Whup",36,"47.4+-25.1","128+-70.3","0.7+-0.2",0,0,0),("Droplet",29,"148+-99.8","252+-120","0.3+-0.16",0,0,0),("Swop",16,"70+-30","214+-85.6","0.3+-0.2",0,0,0)]:
    r(population="N Pacific - SE Alaska (feeding)",region="SE Alaska",call_type=ct,freq_note=f"low freq {lo}; peak {pk}",duration_s=d,n_calls=n,source="Fournet et al. 2018 PeerJ 6:e5365",url="https://peerj.com/articles/5365",verification="full text",notes="SE Alaska vs Massachusetts Bay comparison (Atlantic values not used)")
r(population="N Pacific - SE Alaska (feeding)",region="SE Alaska (cooperative bubble-net groups)",years="1994-1997",call_type="Feeding cry",shape="3 sections: A (onset), B (long near-constant tone), C (end); harmonic, energy to 8 kHz",f_start_hz="A 584 (349-1219); B 594 (349-1203); C 601 (344-1004)",f_end_hz="A 616; B 596 (322-999); C 515 (236-978)",f_peak_hz="B 584 (360-988; median 553)",freq_note="B bandwidth -10 dB 32 (16-65); B freq range median 16 Hz; cv 0.23-0.32",duration_s="2.6 (0.4-8.2); A,C ~0.21; B ~2.3",interval_type="cries per series",interval_s="mean 4.3 (sd 5.1)",n_calls=501,between_encounter_variability="differs among recording sessions/groups (PCA)",source="Cerchio & Dahlheim 2001 Bioacoustics (2001)",url="https://bioacoustics.info/article/variation-feeding-vocalizations-humpback-whales-megaptera-novaeangliae-southeast-alaska",verification="full text (user PDF)")
# Dunlop 2007: type, n, groups, dur, dursd, min, minsd, max, maxsd, start, startsd, end, endsd, peak, peaksd, song
D=[("Wop",64,24,0.748,0.196,43,15,73,16,52,34,73,16,80,19,"no"),("Thwop",67,15,0.949,0.244,42.2,13,83.1,20,43.6,18,83.0,20,94.6,27,"no"),
("Snort",43,17,0.284,0.107,51,25,53,25,53,26,52,25,84,41,"no"),("Grumble",73,20,1.509,1.362,43.6,12,48.5,15,44.2,14,46.7,15,62,24,"no"),
("Sigh",2,1,2.533,0.901,30,10,327,23,183,226,174,193,86,7,"02"),("Siren",5,1,2.309,0.584,95,33,337,48,162,79,272,78,168,18,"02"),
("Short moan",7,5,0.353,0.060,161,34,206,60,168,37,199,64,178,55,"02"),("Horn",18,5,0.365,0.126,180,51,430,116,180,51,401,133,302,32,"02"),
("Violin",12,4,0.312,0.057,548,185,908,239,884,246,616,256,700,152,"02"),("Groan",12,5,1.262,0.630,139,39,172.6,50,156.7,54,160.0,41,156.4,41,"02-04"),
("Ascending moan",4,2,1.854,0.510,162,24,271,29,162,24,233,36,229,38,"02-04"),("Modulated moan",13,5,1.212,0.418,201,24,618,182,206,31,610,188,364,50,"02-04"),
("Cry",8,2,1.623,0.194,338,138,709,202,352,163,627,226,516,89,"no"),("Modulated cry",2,1,2.204,0.517,524,193,1415,32,534,179,1415,32,588,458,"no"),
("Ascending cry",8,1,1.773,0.119,171,46,437,68,259,41,406,79,354,29,"no"),("Trumpet",27,6,0.732,0.345,231,80,854,124,234,77,836,135,471,140,"02"),
("Ascending shriek",2,1,1.252,0.021,714,0,2268,20,714,0,2268,20,953,8,"02-04"),("Descending shriek",7,3,2.297,1.192,809,84,1373,138,889,107,932,90,1083,239,"02"),
("Squeak",6,3,0.213,0.132,1431,302,2420,746,1649,506,2227,897,1478,265,"02-04"),
("Purr",16,4,1.566,0.575,56,43,59,41.8,56,43.9,59,41.8,206,62,"02"),("Growl",14,5,2.253,1.268,60,18,73,17,62,18,70,18,73,36,"02-04"),
("Trill",6,3,2.194,0.468,245,83,427,92,261,72,342,127,360,42,"02"),("Blow",6,3,0.281,0.152,45.5,15,45.5,15,45.5,15,46.3,20,191,57,"no"),
("Bark",50,13,0.205,0.119,139,98,346.9,204,153.4,104,350.4,201,225.7,106,"02"),("Bellow",4,4,0.939,0.415,382,117,403,100,379,114,404,100,437,97,"no"),
("Creak",4,1,1.723,0.614,29,7,102,26,29,7,102,26,243,15,"02"),("Screech",4,1,1.366,0.336,119,42,221,39,125,51,216.0,45,365,38,"02"),
("Scream",8,3,1.179,0.288,678,156,1436,270,684,149,1157,248,1177,166,"no"),("Grunts",90,9,0.221,0.110,43,20,45,23,43,20,43,20,74.8,39,"03-04"),
("Croaks",17,3,0.522,0.255,75,45,120,61,75,45,120,61,193,62,"02"),("Yap",10,4,0.372,0.101,230,94,2480,875,2480,875,250,94,1325,611,"02"),
("Yelps",21,1,0.251,0.074,95,42,372,158,236,118,173,111,170,46,"no"),("Pulses",25,2,0.062,0.027,139,11,294,152,191,102,188,39,171,63,"no"),
("Low yap",5,1,0.129,0.037,243,128,820,310,712,85,268,383,488,330,"no")]
pulse={"Purr":"21.6+-6.1","Growl":"67.3+-3.9","Trill":"40.5+-3.4"}
bout={"Grunts":"5","Croaks":"4-5 (3-8)","Yap":"6-8","Yelps":"3-4","Pulses":"7-15","Low yap":"6"}
for (ct,n,g,d,dsd,mn,mnsd,mx,mxsd,st,stsd,en,ensd,pk,pksd,song) in D:
    r(population="S Pacific - E Australia (migrating)",region="Peregian Beach, E Australia",years="2002-2004",call_type=ct,shape="see Dunlop 2007 Tables II-III",f_start_hz=st,f_start_sd=stsd,f_end_hz=en,f_end_sd=ensd,f_peak_hz=pk,f_peak_sd=pksd,
      freq_note=f"F0 min {mn} (SD {mnsd}), max {mx} (SD {mxsd})",pulse_rate_hz=pulse.get(ct,""),duration_s=d,duration_sd=dsd,interval_type=("units per bout" if ct in bout else ""),interval_s=bout.get(ct,""),
      n_calls=n,n_animals_or_sites=f"{g} groups",source="Dunlop et al. 2007 JASA 122(5)",url="(user PDF) Dunlop et al. 2007",verification="full text (user PDF); wide table, column alignment checked by eye",
      notes=("also a song unit in "+song if song!="no" else "not a song unit"))
r(population="S Pacific - E Australia (migrating)",region="E Australia",years="1997, 2003-04, 2008",call_type="Repertoire stability (mixture weights)",freq_note="12 stable types = 64.2% of calls: cry 51, asc moan 25, bark 63, grumble 160, grunt 147, mod moan 37, thwop 87, snort 110, whoop 38, wop 134, yap 105, yelp 28 (calls over 3 periods); song-unit calls = 23.2% (change with song); inconsistent types = 12.6%",between_encounter_variability="year-to-year: song-unit calls appear/disappear with song; stable core persists 11 yr",source="Rekdahl et al. 2013 JASA (2013)",url="(user PDF) Rekdahl et al. 2013",verification="full text (user PDF)",notes="use as call-type weights for social-call encounters")
r(population="All Pacific (song)",call_type="Song units (generic)",freq_note="song changes yearly; units span ~20 Hz to >4 kHz; E Australian song-unit social calls (Dunlop 2007 rows flagged as song units) are a reasonable unit pool",source="Dunlop et al. 2007; Rekdahl et al. 2013",verification="synthesis",notes="simulate song as random sequences of these unit types with phrase repetition; not a population-specific song")
write(R,"Humpback","HB","literature_call_parameters_humpback.csv")
