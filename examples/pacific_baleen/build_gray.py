from build_common import write
R=[]
def r(**k): R.append(k)
BU="https://www.frontiersin.org/articles/10.3389/fmars.2018.00329/pdf"
CL="(user PDF) Crane & Lashkari 1996 JASA 100(3)"
DA="(user PDF) Dahlheim 1987 PhD thesis, UBC"
# Burnham 2018: class, dir, n, low, lowsd, high, highsd, peak, peaksd, dur, dursd
B=[("1a Modulated knock","N",994,174.33,140.21,760.34,428.89,258.34,82.00,1.58,1.08,"pulsed knock series"),
("1a Modulated knock","S",34,19.75,12.18,214.03,136.28,65.44,51.80,2.40,2.57,"pulsed knock series"),
("1b Unmodulated knock","N",21,42.43,91.60,219.68,251.98,63.64,99.24,2.32,0.85,"pulsed knock series"),
("1b Unmodulated knock","S",8,99.04,93.41,406.25,171.34,197.89,128.61,3.03,1.35,"pulsed knock series"),
("2a Upsweep","N",998,43.58,26.60,179.93,68.23,79.34,39.69,1.65,0.59,"tonal upsweep"),
("2a Upsweep","S",26,33.43,20.19,159.10,26.45,68.55,30.72,2.78,1.21,"tonal upsweep"),
("2b Downsweep","N",65,30.81,17.27,123.02,45.39,59.88,25.69,2.21,0.67,"tonal downsweep (Table 2 lists 143 Hz / 0.89 s - inconsistent)"),
("2b Downsweep","S",4,27.03,27.36,116.55,56.44,53.00,31.09,5.09,1.50,"tonal downsweep"),
("3 Moan (M3)","N",11506,19.58,15.66,124.62,33.44,48.03,24.55,2.14,0.80,"low-frequency moan, AM/pulsed"),
("3 Moan (M3)","S",3619,16.09,7.49,114.27,31.38,48.73,25.34,2.52,1.14,"low-frequency moan"),
("3a Low moan","N",172,32.36,13.27,58.61,20.50,44.90,15.64,2.27,0.94,"narrow low moan"),
("4 Grunt/rumble","N",19,36.32,25.78,151.44,67.65,66.09,30.82,2.37,0.84,"grunt")]
for ct,dr,n,lo,losd,hi,hisd,pk,pksd,d,dsd,shape in B:
    r(population="Eastern N Pacific (migrating)",region="W Vancouver Island (Flores Island)",years="2015-2017",call_type=ct,unit=("northbound" if dr=="N" else "southbound"),shape=shape,
      f_start_hz=f"(low) {lo}",f_start_sd=losd,f_end_hz=f"(high) {hi}",f_end_sd=hisd,f_peak_hz=pk,f_peak_sd=pksd,duration_s=d,duration_sd=dsd,n_calls=n,
      between_encounter_variability="direction (N vs S migration) shifts call mix: moans 84% N, 98% S",source="Burnham, Duffus & Mouy 2018 Front Mar Sci 5:329",url=BU,verification="full text",
      notes="low/high = min/max freq, not start/end",include_v1=("no - not tonal (knock)" if "knock" in ct else "yes"))
r(population="Eastern N Pacific (migrating)",region="Monterey & Carmel Bay, California",years="1988-1991",call_type="M3 low-frequency moan",shape="AM moan, slow modulation of centre frequency, sidebands",f_peak_hz="centre 74.0",f_peak_sd=19.4,freq_note="frequency range 142.5+-97.6 (12.5-550); 3 dB bandwidth 11.6+-4.6; centre range 40-132.5; sideband interval 12.95+-7.48 Hz (7.5-49); Q 7.0",duration_s=1.14,duration_sd=0.43,n_calls="61 (46.6% of 258)",source="Crane & Lashkari 1996 JASA",url=CL,verification="full text (user PDF, OCR checked against page image)",notes="likely energy below 40 Hz recording limit")
r(population="Eastern N Pacific (migrating)",region="Monterey & Carmel Bay",years="1988-1991",call_type="M4 grunt",shape="moan with zipper-like pulse structure",f_peak_hz="centre 121.67",f_peak_sd=62.02,freq_note="range 509.1+-254.2 (147-1000); 3 dB bw 68.3+-35.5",duration_s=0.70,duration_sd=0.35,n_calls=6,source="Crane & Lashkari 1996",url=CL,verification="full text (user PDF)")
r(population="Eastern N Pacific (migrating)",region="Monterey & Carmel Bay",years="1988-1991",call_type="M1 clicks / knocks / bongs",shape="pulse groups, variable centre freq, harmonic within pulses",f_peak_hz="centre 665.5",f_peak_sd=480.8,freq_note="range 887.4+-858.6 (45-4520); 3 dB bw 141+-95; pulse duration 0.043+-0.035 s; 5.2+-4.2 pulses (1-18)",duration_s=0.911,duration_sd=0.88,n_calls="49 (37.4%)",source="Crane & Lashkari 1996",url=CL,verification="full text (user PDF)",include_v1="no - not tonal (knock)")
r(population="Eastern N Pacific (migrating)",region="Monterey & Carmel Bay",call_type="M5 exhalation / bubble blast",freq_note="centre 127.7+-45.2; wideband",duration_s=2.53,duration_sd=1.73,n_calls=15,source="Crane & Lashkari 1996",url=CL,verification="full text (user PDF)",include_v1="no - non-vocal")
# Dahlheim lagoon (S) and Arctic (N)
r(population="Eastern N Pacific (breeding lagoon)",region="Laguna San Ignacio, Mexico",years="1982-1984",call_type="S1 knock/pulse series",shape="pulse series with FM",freq_note="range 90 (120-1250) to 1940 (1000-3000); energy 332-824; 9.4 pulses/series (2-30); 5.9 pulses/s (2.2-14.7)",duration_s="1.8 (0.3-4.6)",n_calls=1000,source="Dahlheim 1987 PhD thesis (Table II)",url=DA,verification="full text (user PDF); mean (range)",notes="most common lagoon call",include_v1="no - not tonal (knock)")
r(population="Eastern N Pacific (breeding lagoon)",region="Laguna San Ignacio",call_type="S2",freq_note="250-300 Hz; energy 250-300",duration_s="0.3 (0.2-0.4)",n_calls=100,source="Dahlheim 1987",url=DA,verification="full text (user PDF)",notes="5% of calls; not heard on migration")
r(population="Eastern N Pacific (breeding lagoon)",region="Laguna San Ignacio",call_type="S3 moan",shape="pulse + frequency modulated moan",freq_note="125 (80-200) to 1250 (750-1800); energy 170-430",duration_s="2.0 (1.0-4.0)",n_calls=100,source="Dahlheim 1987",url=DA,verification="full text (user PDF)",notes="1% of lagoon calls")
r(population="Eastern N Pacific (breeding lagoon)",region="Laguna San Ignacio",call_type="S4 grunt",shape="pulse modulated",freq_note="150 (125-200) to 1570 (1500-1600); energy 225-600",duration_s="0.9 (0.7-1.4)",n_calls=100,source="Dahlheim 1987",url=DA,verification="full text (user PDF)")
r(population="Eastern N Pacific (breeding lagoon)",region="Laguna San Ignacio",call_type="S5 bubble blast / S6 exhalation",freq_note="S5 130-840 (energy <500), 3.2 s; S6 250-850 (<700), 3.3 s",n_calls="100 each",source="Dahlheim 1987",url=DA,verification="full text (user PDF)",include_v1="no - non-vocal")
r(population="Eastern N Pacific (Arctic feeding)",region="Bering/Chukchi (Moore & Ljungblad 1984)",call_type="N1 pulse series",freq_note="200-4000 Hz; energy 963 (238-2584); 12 pulses/series (2-69); 7.1 pulses/s; no FM",duration_s="1.4 (0.4-2.3)",source="Moore & Ljungblad 1984",source_via="Dahlheim 1987 Table IV",url=DA,verification="secondary",include_v1="no - not tonal (knock)")
r(population="Eastern N Pacific (Arctic feeding)",region="Bering/Chukchi",call_type="N3 moan",freq_note="102-578 Hz; energy 325",duration_s=0.9,source="Moore & Ljungblad 1984",source_via="Dahlheim 1987",url=DA,verification="secondary")
r(population="Eastern N Pacific (Arctic feeding)",region="Bering/Chukchi",call_type="N4 grunt",freq_note="136-748 Hz; energy 388",duration_s="0.3 (0.2-0.5)",source="Moore & Ljungblad 1984",source_via="Dahlheim 1987",url=DA,verification="secondary")
r(population="Eastern N Pacific (migrating)",region="California (Cummings 1968)",call_type="M3 moan",freq_note="20-200 Hz",duration_s=1.4,source="Cummings et al. 1968",source_via="Dahlheim 1987 Table IV",url=DA,verification="secondary")
r(population="Eastern N Pacific (migrating)",region="California",call_type="M3 moan (Guazzo)",f_peak_hz=38.1,duration_s=1.79,freq_note="moans 22% of calls",source="Guazzo et al. 2017 PLoS ONE",source_via="Burnham 2018 Table 2",url=BU,verification="secondary")
write(R,"Gray","GW","literature_call_parameters_gray.csv")
