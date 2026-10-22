# T7 examples: real errors of the frozen gate on the clean test split

Two seeded-random (seed 0) examples per bucket per error type. Texts are the corpus text, first 200 characters (whitespace collapsed). `p` = fused calibrated machine-probability; headA / headB = the two raw head scores (higher = more machine-like, headB is inverted on te/hi, see decisions.md). Decision support only: these are the detector's mistakes, not verdicts on authors.

## en  (tau 0.05 = 0.707, tau 0.01 = 0.995, HUMAN cutoff = 0.029)

### False flag (human text flagged MACHINE at alpha 0.05) — 21 such rows

- `en_wikipedia_1313` — wikipedia, 169 words, p = 0.927, headA = 0.679, headB = -1.185, also flagged at alpha 0.01: no
  > Aromatic compounds, also known as "mono- and polycyclic aromatic hydrocarbons", are organic compounds containing one or more aromatic rings. The word "aromatic" originates from the past grouping of mo

- `en_samanantar_en_hi:2617-2627` — samanantar_en, 191 words, p = 0.992, headA = 0.949, headB = -2.221, also flagged at alpha 0.01: no
  > Apart from Gandhi, party leader Adhir Ranjan Chaudhary, party's chief whip Lok Sabha Kodikunnil Suresh among other MPs participated in the protest. The new office, with its high walls, enclosed struct

### False clear (machine text called HUMAN) — 2 such rows

- `qwen7b__en_hc3_wiki_csai:286` — qwen7b, 270 words, p = 0.023, headA = 0.014, headB = -2.986
  > The concept of integrating a non-negative function to calculate the area under its curve is a fundamental pillar in calculus with far-reaching implications across various fields such as physics, engin

- `mistral__en_hc3_wiki_csai:353` — mistral, 276 words, p = 0.027, headA = 0.000, headB = -0.707
  > A vector space, as the name suggests, is a structured mathematical environment where vectors can be manipulated according to two fundamental operations: addition and scalar multiplication. The set of 

## hi  (tau 0.05 = 0.737, tau 0.01 = 0.974, HUMAN cutoff = 0.023)

### False flag (human text flagged MACHINE at alpha 0.05) — 20 such rows

- `hi_indiccorp_v2_hin_Deva:1404` — indiccorp_v2, 165 words, p = 0.858, headA = 0.726, headB = -1.188, also flagged at alpha 0.01: no
  > 1. के एक आश्चर्यजनक चयन प्रदान करता है सभी प्रकार के ग्राहकों के लिए मुट्ठी ट्रॉफी। इन वस्तुओं को विभिन्न प्रकार के कपड़ों पर पिन किया जा सकता है और यह किसी के व्यक्तित्व और प्रदर्शन के हितों को व्यक्

- `hi_indiccorp_v2_hin_Deva:11050` — indiccorp_v2, 125 words, p = 0.764, headA = 0.654, headB = -1.698, also flagged at alpha 0.01: no
  > लू 16वीं सदी में पहली बार यूरोप पहुंचा था और उसकी पकड़ बहुत धीमी थी। कई ताकतों ने इसका विरोध भी किया था, जिसमें पादरी भी शामिल थे। उन्होंने आलू खाने से इसलिए मना किया था, क्योंकि बाइबिल में आलू का उल्ल

### False clear (machine text called HUMAN) — 7 such rows

- `llama__hi_indiccorp_v2_hin_Deva:27708` — llama, 131 words, p = 0.022, headA = 0.041, headB = -2.959
  > आगमन पश्चात सतनाम संदेश यात्रा के नेतृत्वकर्ता बबलू त्रिवेंद्र रायपुर शहर अध्य, मोहन बंजारे प्रदेश अध्यक्ष प्रगतिशील छत्तीसगढ़ सतनामी समाज युवा प्रकोष्ठ, दीपक मिरी रायपुर जिला अध्यक्ष ग्रामीण एवं सम्मा

- `qwen7b__hi_indiccorp_v2_hin_Deva:42308` — qwen7b, 108 words, p = 0.012, headA = 0.003, headB = -6.110
  > आष्टक शब्दांची एक महत्वाची शिक्षण संरचना होती जे त्याला स्मृति, कल्पना, भावना स्थूलाकार करतात. मजबूत शिक्षण प्रक्रियेत एष्टक बहुत परिचालित झाले अनेक स्थितींमध्ये. पद्य साहित्यात, गद्यात आणि कवितांसाठी

## te  (tau 0.05 = 0.417, tau 0.01 = 0.983, HUMAN cutoff = 0.005)

### False flag (human text flagged MACHINE at alpha 0.05) — 20 such rows

- `te_wikipedia_1127` — wikipedia, 278 words, p = 0.967, headA = 0.907, headB = -6.316, also flagged at alpha 0.01: no
  > పద్య పాదంలో మొదటి అక్షరానికి ఆ పద్యం లక్షణములలో చెప్పబడిన యతి స్థానంలో మైత్రి గల అక్షరాన్ని వాడటాన్నే "యతి మైత్రి" అంటారు. యతి అంటే విరామం అని అర్థం. లయబద్ధమైన పద్య నడకలో సహజంగా వచ్చే విరామాన్ని యతి స

- `te_indiccorp_v2_tel_Telu:49734` — indiccorp_v2, 122 words, p = 0.988, headA = 0.973, headB = -3.742, also flagged at alpha 0.01: yes
  > చివరగా, శాశ్వతత్వం తాకే బాధ, ఒక ఇటాలియన్ ఒక ప్రయాణిస్తున్న అపరిచితుల ఆకర్షించింది. కొన్ని తెలియని కారణంగా, అతను ఉండాలని నిర్ణయించుకున్నాడు, మరియు మా రోమ్ యొక్క అహంకారం చూడటానికి అవకాశం వచ్చింది. విదేశ

### False clear (machine text called HUMAN) — 3 such rows

- `gemma__te_indiccorp_v2_tel_Telu:64994` — gemma, 136 words, p = 0.003, headA = 0.000, headB = -6.368
  > నాసా వెలుగులో మరో 200 గ్రహాలు! భూమికి సన్నని పోలిక న్యూయార్క్, జూన్ 20: విశ్వంలో భూమిని పోలిన గ్రహాల అనేవషణను నిరంతరం సాగిస్తున్న అమెరికా అంతరిక్ష పరిశోధన సంస్థ(నాసా) తాజాగా మలో 200 కొత్త గ్రహాలను కను

- `qwen7b__te_indiccorp_v2_tel_Telu:121438` — qwen7b, 208 words, p = 0.001, headA = 0.018, headB = -17.153
  > న్యూఢిల్లీ మే 2 న, ప్రధాన మంత్రి నరేంద్ర మోదీ నాయకత్వంలోని ఎన్డీయే ప్రభుత్వం సీనియర్ సిటిజన్ జీవితదారులకు అత్యంత మేలు కలిగించే నిర్ణయం తీసి వారికి ప్రదర్శన రోజులను, స్విచ్‌ప్లాన్స్ మరియు ఉద్యోగ కష్టం 

## cm  (tau 0.05 = 0.819, tau 0.01 = 0.871, HUMAN cutoff = 0.133)

### False flag (human text flagged MACHINE at alpha 0.05) — 20 such rows

- `cm_comi_lingua_TN:train:2417` — comi_lingua, 16 words, p = 0.893, headA = 0.978, headB = 0.271, also flagged at alpha 0.01: yes
  > Lagta hai kuch chamche poora video dekhte bhi nahi honge, usse pehle dislike kar dete hain.

- `cm_comi_lingua_TN:train:2444` — comi_lingua, 16 words, p = 0.853, headA = 0.931, headB = -0.290, also flagged at alpha 0.01: no
  > Jaati par chal raha hai, auqat pe nahi. Sir ji, nirikshan karwayein tab samajh mein aayega.

### False clear (machine text called HUMAN) — 22 such rows

- `gemma__cm_comi_lingua_TN:train:1741` — gemma, 17 words, p = 0.031, headA = 0.161, headB = 0.163
  > Ayushman card to bana hua hai lekin isase ilag nahi hota. Kisi kam ka bhi nahi hai.

- `qwen7b__cm_comi_lingua_TN:train:27` — qwen7b, 18 words, p = 0.093, headA = 0.338, headB = -0.278
  > isko kya jawab hain? Woh to sabhi ko samajhne ke bandha tha, nahi to kya!isko kya jawab hain?

