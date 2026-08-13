# Specificity run — inspection bundle

2 question(s), 12 cell(s). Gold is FIXED across levels (the guardrail); level 0 = ambiguous, level 1 = disambiguated.

## 1. `5780388869788119926` — m0=3, target_idx=1, model `qwen_2_5_7b`
**Fixed gold (a_i variants):** Nkosazana Clarice Dlamini-Zuma, Nkosazana Dlamini-Zuma

**Uniform evidence** (16 snippets, identical for both levels & all paraphrases); first entries:
- Chairperson of the African Union Commission: The Chairperson of the African Union Commission is the head of the African Union Comm…
- African Union Commission: The Commission of the African Union acts as the executive/administrative branch or secretariat ... Mouss…
- Chairperson of the African Union: The Chairperson of the African Union is the ceremonial head of the African Union elected by ... …
- African Union: Selection of the chairperson[edit]. In 2006, the AU decided to create a Committee "to consider the implementation o…

### Level 0 — “Who is the current chairman of african union commission?”
FI_spec=0.000 · f_mean=0.000 · AUFI_in=3.373 · FI_out=0.812 · H_sem=0.188 · |A_q|=2

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Who presently serves as the chairperson of the African Union Commission? | Moussa Faki | 0 |
| 1 | Who presently holds the position of chairperson for the African Union Commission… | Moussa Faki | 0 |
| 2 | Current chairman African Union Commission who? | Moussa Faki | 0 |
| 3 | Who currently serves as the Chairman of the African Union Commission? | Moussa Faki | 0 |
| 4 | Who presently holds the position of chairman of the African Union Commission? | Moussa Faki | 0 |
| 5 | Who holds the position of Chairman of the African Union Commission at present? | Moussa Faki | 0 |
| 6 | Who presently occupies the position of Chairman of the African Union Commission? | Moussa Faki | 0 |
| 7 | Who's the current chairman of the African Union Commission? | Moussa Faki | 0 |
| 8 | Who presently holds the office of the Chairman of the African Union Commission? | Moussa Faki | 0 |
| 9 | Who's the chairman of the African Union Commission right now? | Moussa Faki | 0 |

H_sem (k=10): 2 clusters — c0: “Moussa Faki” · c1: “Moussa Faki Mahamat”

### Level 0 — “Who is the current chairman of african union commission?”
FI_spec=0.000 · f_mean=0.000 · AUFI_in=3.373 · FI_out=1.567 · H_sem=1.018 · |A_q|=6

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Who presently serves as the chairperson of the African Union Commission? | Moussa Faki Mahamat. | 0 |
| 1 | Who presently holds the position of chairperson for the African Union Commission… | Moussa Faki Mahamat. | 0 |
| 2 | Current chairman African Union Commission who? | Moussa Faki Mahamat. | 0 |
| 3 | Who currently serves as the Chairman of the African Union Commission? | Moussa Faki Mahamat. | 0 |
| 4 | Who presently holds the position of chairman of the African Union Commission? | Moussa Faki Mahamat. | 0 |
| 5 | Who holds the position of Chairman of the African Union Commission at present? | Moussa Faki Mahamat. | 0 |
| 6 | Who presently occupies the position of Chairman of the African Union Commission? | Moussa Faki Mahamat. | 0 |
| 7 | Who's the current chairman of the African Union Commission? | Moussa Faki Mahamat. | 0 |
| 8 | Who presently holds the office of the Chairman of the African Union Commission? | Moussa Faki Mahamat. | 0 |
| 9 | Who's the chairman of the African Union Commission right now? | Moussa Faki Mahamat. | 0 |

H_sem (k=10): 6 clusters — c0: “Moussa Faki.” · c1: “Moussa Faki Mahamat.” · c2: “Moussa Faki Mahamat, unknown title, (hea…” · c3: “Moussa Faki Mahamat since 14 March 2017.” · c4: “Moussa Faki Mahamat at present.” · c5: “Moussa Faki is Chairman of the Commissio…”

### Level 0 — “Who is the current chairman of african union commission?”
FI_spec=0.000 · f_mean=0.000 · AUFI_in=3.373 · FI_out=1.114 · H_sem=0.471 · |A_q|=3

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Who presently serves as the chairperson of the African Union Commission? | Moussa Faki | 0 |
| 1 | Who presently holds the position of chairperson for the African Union Commission… | Moussa Faki | 0 |
| 2 | Current chairman African Union Commission who? | Moussa Faki | 0 |
| 3 | Who currently serves as the Chairman of the African Union Commission? | Moussa Faki | 0 |
| 4 | Who presently holds the position of chairman of the African Union Commission? | Moussa Faki | 0 |
| 5 | Who holds the position of Chairman of the African Union Commission at present? | Moussa Faki | 0 |
| 6 | Who presently occupies the position of Chairman of the African Union Commission? | Moussa Faki | 0 |
| 7 | Who's the current chairman of the African Union Commission? | Moussa Faki | 0 |
| 8 | Who presently holds the office of the Chairman of the African Union Commission? | Moussa Faki | 0 |
| 9 | Who's the chairman of the African Union Commission right now? | Moussa Faki | 0 |

H_sem (k=10): 3 clusters — c0: “Moussa Faki” · c1: “Moussa Faki Mahamat” · c2: “Moussa Faki (current)”

### Level 1 — “Who is the 3rd chairman of african union commission?”
FI_spec=1.585 · f_mean=0.000 · AUFI_in=3.373 · FI_out=1.611 · H_sem=0.711 · |A_q|=5

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Who is the third chairperson of the African Union Commission? | Moussa Faki | 0 |
| 1 | Who's the 3rd chairman of the African Union Commission? | unknown | 0 |
| 2 | Who serves as the third Chairperson of the African Union Commission? | Moussa Faki | 0 |
| 3 | Who holds the position of third chairperson of the African Union Commission? | unknown | 0 |
| 4 | Who holds the position of the third Chairman of the African Union Commission? | unknown | 0 |
| 5 | What is the identity of the third Chairperson of the African Union Commission? | Moussa Faki | 0 |
| 6 | Who's the third person to chair the African Union Commission? | unknown | 0 |
| 7 | Which individual occupies the position of the third chairperson of the African U… | Moussa Faki | 0 |
| 8 | Who is the third person to be chairman of the African Union Commission? | unknown | 0 |
| 9 | Who's the third AU Commission chairman? | unknown | 0 |

H_sem (k=10): 5 clusters — c0: “Moussa Faki” · c1: “unknown” · c2: “Jean Ping” · c3: “Moussa Faki Mahamat” · c4: “Nkosazana Dlamini-Zuma”

### Level 1 — “Who is the 3rd chairman of african union commission?”
FI_spec=1.585 · f_mean=0.000 · AUFI_in=3.373 · FI_out=2.599 · H_sem=2.101 · |A_q|=26

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Who is the third chairperson of the African Union Commission? | unknown | 0 |
| 1 | Who's the 3rd chairman of the African Union Commission? | unknown | 0 |
| 2 | Who serves as the third Chairperson of the African Union Commission? | Moussa Faki. | 0 |
| 3 | Who holds the position of third chairperson of the African Union Commission? | unknown | 0 |
| 4 | Who holds the position of the third Chairman of the African Union Commission? | Jean Ping. | 0 |
| 5 | What is the identity of the third Chairperson of the African Union Commission? | unknown | 0 |
| 6 | Who's the third person to chair the African Union Commission? | Jean Ping. | 0 |
| 7 | Which individual occupies the position of the third chairperson of the African U… | Jean Ping. | 0 |
| 8 | Who is the third person to be chairman of the African Union Commission? | unknown | 0 |
| 9 | Who's the third AU Commission chairman? | unknown | 0 |

H_sem (k=10): 26 clusters — c0: “Jean Ping.” · c1: “Jean Ping, Ping was the third to hold th…” · c10: “Nkosazana Dlamini-Zuma was the third Cha…” · c11: “Moussa Faki.” · c12: “Moussa Faki was the third Chairman.” · c13: “Jean Ping, previously and then Nkosazana…” · c14: “Nkosazana Dlamini-Zuma then Moussa Faki …” · c15: “Nkosazana Dlamini-Zuma is not the third.”

### Level 1 — “Who is the 3rd chairman of african union commission?”
FI_spec=1.585 · f_mean=0.000 · AUFI_in=3.373 · FI_out=2.804 · H_sem=0.518 · |A_q|=10

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Who is the third chairperson of the African Union Commission? | Moussa Faki | 0 |
| 1 | Who's the 3rd chairman of the African Union Commission? | Moussa Faki | 0 |
| 2 | Who serves as the third Chairperson of the African Union Commission? | Moussa Faki | 0 |
| 3 | Who holds the position of third chairperson of the African Union Commission? | Moussa Faki (unknown if he is the third chairperson specifically, as the context… | 0 |
| 4 | Who holds the position of the third Chairman of the African Union Commission? | Moussa Faki | 0 |
| 5 | What is the identity of the third Chairperson of the African Union Commission? | Moussa Faki | 0 |
| 6 | Who's the third person to chair the African Union Commission? | Moussa Faki | 0 |
| 7 | Which individual occupies the position of the third chairperson of the African U… | Moussa Faki | 0 |
| 8 | Who is the third person to be chairman of the African Union Commission? | Moussa Faki | 0 |
| 9 | Who's the third AU Commission chairman? | Moussa Faki | 0 |

H_sem (k=10): 10 clusters — c0: “Moussa Faki” · c1: “Moussa Faki (since January 2017)” · c2: “Moussa Faki (2017 - present)” · c3: “Moussa Faki (unknown if a "third" as not…” · c4: “Moussa Faki (January 30, 2017)” · c5: “Moussa Faki (given context does not spec…” · c6: “Moussa Faki (Third term starts in Januar…” · c7: “Moussa Faki (third after Jean Ping and N…”

## 2. `-196855123397981737` — m0=3, target_idx=0, model `qwen_2_5_7b`
**Fixed gold (a_i variants):** National Fascist Party

**Uniform evidence** (20 snippets, identical for both levels & all paraphrases); first entries:
- Fascist Italy (1922–1943): In 1929, education policy took a major step towards being completely taken over by the agenda of indoct…
- Kingdom of Italy: The Kingdom of Italy (Italian: Regno d'Italia) was a state which existed from 1861 —when King ... "Fascist Italy…
- 1922 in Italy: The Italian government refuses to take over the powers in the city and provisionally entrusts it to a military comm…
- Economy of Italy under fascism: The economy of Italy under fascism refers to the economy in the Kingdom of Italy between 1922 and …

### Level 0 — “Who took control of the italian government in 1922?”
FI_spec=0.000 · f_mean=0.000 · AUFI_in=3.373 · FI_out=0.000 · H_sem=0.000 · |A_q|=1

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | In 1922, who assumed control of the Italian government? | Benito Mussolini | 0 |
| 1 | Who took over Italy's government in 1922? | Benito Mussolini | 0 |
| 2 | Who took control of Italy's government in the year 1922? | Benito Mussolini | 0 |
| 3 | Who assumed control of the Italian government in 1922? | Benito Mussolini | 0 |
| 4 | Who took over the Italian government in 1922? | Benito Mussolini | 0 |
| 5 | Who assumed control of the Italian government in the year 1922? | Benito Mussolini | 0 |
| 6 | In 1922, who became in charge of the government of Italy? | Benito Mussolini | 0 |
| 7 | Who seized control of Italy's government in 1922? | Benito Mussolini | 0 |
| 8 | Who grabbed power in Italy's government back in 1922? | Benito Mussolini | 0 |
| 9 | Who assumed authority over the Italian government in the year 1922? | Benito Mussolini | 0 |

H_sem (k=10): 1 clusters — c0: “Benito Mussolini”

### Level 0 — “Who took control of the italian government in 1922?”
FI_spec=0.000 · f_mean=0.000 · AUFI_in=3.373 · FI_out=2.283 · H_sem=0.302 · |A_q|=6

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | In 1922, who assumed control of the Italian government? | Benito Mussolini. | 0 |
| 1 | Who took over Italy's government in 1922? | Benito Mussolini. | 0 |
| 2 | Who took control of Italy's government in the year 1922? | Benito Mussolini. | 0 |
| 3 | Who assumed control of the Italian government in 1922? | Benito Mussolini. | 0 |
| 4 | Who took over the Italian government in 1922? | Benito Mussolini. | 0 |
| 5 | Who assumed control of the Italian government in the year 1922? | Benito Mussolini. | 0 |
| 6 | In 1922, who became in charge of the government of Italy? | Benito Mussolini. | 0 |
| 7 | Who seized control of Italy's government in 1922? | Benito Mussolini. | 0 |
| 8 | Who grabbed power in Italy's government back in 1922? | Benito Mussolini. | 0 |
| 9 | Who assumed authority over the Italian government in the year 1922? | Benito Mussolini. | 0 |

H_sem (k=10): 6 clusters — c0: “Benito Mussolini.” · c1: “Benito Mussolini took over Italy's gover…” · c2: “Benito Mussolini took over Italy's gover…” · c3: “Benito Mussolini and the Fascists.” · c4: “Benito Mussolini took control of Italy's…” · c5: “Benito Mussolini took control of the gov…”

### Level 0 — “Who took control of the italian government in 1922?”
FI_spec=0.000 · f_mean=0.000 · AUFI_in=3.373 · FI_out=0.953 · H_sem=0.047 · |A_q|=2

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | In 1922, who assumed control of the Italian government? | Benito Mussolini | 0 |
| 1 | Who took over Italy's government in 1922? | Benito Mussolini | 0 |
| 2 | Who took control of Italy's government in the year 1922? | Benito Mussolini | 0 |
| 3 | Who assumed control of the Italian government in 1922? | Benito Mussolini | 0 |
| 4 | Who took over the Italian government in 1922? | Benito Mussolini | 0 |
| 5 | Who assumed control of the Italian government in the year 1922? | Benito Mussolini | 0 |
| 6 | In 1922, who became in charge of the government of Italy? | Benito Mussolini | 0 |
| 7 | Who seized control of Italy's government in 1922? | Benito Mussolini | 0 |
| 8 | Who grabbed power in Italy's government back in 1922? | Benito Mussolini | 0 |
| 9 | Who assumed authority over the Italian government in the year 1922? | Benito Mussolini | 0 |

H_sem (k=10): 2 clusters — c0: “Benito Mussolini” · c1: “Fascists (or the National Fascist Party)”

### Level 1 — “What party took control of the Italian government in 1922?”
FI_spec=1.585 · f_mean=0.900 · AUFI_in=0.148 · FI_out=0.903 · H_sem=0.097 · |A_q|=2

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Which party took over the Italian government in 1922? | National Fascist Party | 1 |
| 1 | Which party took over Italy's government in 1922? | National Fascist Party | 1 |
| 2 | Which party became in charge of Italy's government in 1922? | National Fascist Party | 1 |
| 3 | Which party grabbed power in Italy's government back in 1922? | National Fascist Party | 1 |
| 4 | Which party became in charge of the Italian government in 1922? | National Fascist Party | 1 |
| 5 | Which party did take control of the government in Italy in the year 1922? | National Fascist Party | 1 |
| 6 | Which party became in charge of the government in Italy in the year 1922? | National Fascist Party | 1 |
| 7 | Which party became in charge of the Italian government in the year 1922? | National Fascist Party | 1 |
| 8 | Which political party assumed governance of Italy in the year 1922? | National Fascist Party | 1 |
| 9 | Which political faction assumed governmental authority in Italy during 1922? | Fascist faction | 0 |

H_sem (k=10): 2 clusters — c0: “National Fascist Party” · c1: “Fascist faction”

### Level 1 — “What party took control of the Italian government in 1922?”
FI_spec=1.585 · f_mean=0.800 · AUFI_in=0.314 · FI_out=1.799 · H_sem=1.009 · |A_q|=7

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Which party took over the Italian government in 1922? | The National Fascist Party. | 1 |
| 1 | Which party took over Italy's government in 1922? | The National Fascist Party. | 1 |
| 2 | Which party became in charge of Italy's government in 1922? | The National Fascist Party. | 1 |
| 3 | Which party grabbed power in Italy's government back in 1922? | The National Fascist Party, led by Benito Mussolini. | 0 |
| 4 | Which party became in charge of the Italian government in 1922? | The National Fascist Party. | 1 |
| 5 | Which party did take control of the government in Italy in the year 1922? | The National Fascist Party. | 1 |
| 6 | Which party became in charge of the government in Italy in the year 1922? | The National Fascist Party. | 1 |
| 7 | Which party became in charge of the Italian government in the year 1922? | The National Fascist Party. | 1 |
| 8 | Which political party assumed governance of Italy in the year 1922? | The National Fascist Party. | 1 |
| 9 | Which political faction assumed governmental authority in Italy during 1922? | The Fascists, led by Benito Mussolini. | 0 |

H_sem (k=10): 7 clusters — c0: “National Fascist Party.” · c1: “The Fascist Party, led by Benito Mussoli…” · c2: “The National Fascist Party under Benito …” · c3: “National Fascist Party under Benito Muss…” · c4: “National Fascist Party government.” · c5: “The Fascists.” · c6: “The Fascist regime and Italian Fascists.”

### Level 1 — “What party took control of the Italian government in 1922?”
FI_spec=1.585 · f_mean=1.000 · AUFI_in=0.000 · FI_out=1.466 · H_sem=0.119 · |A_q|=3

| # | paraphrase | model answer (T=0) | F |
|--:|------------|--------------------|--:|
| 0 | Which party took over the Italian government in 1922? | National Fascist Party | 1 |
| 1 | Which party took over Italy's government in 1922? | National Fascist Party | 1 |
| 2 | Which party became in charge of Italy's government in 1922? | National Fascist Party | 1 |
| 3 | Which party grabbed power in Italy's government back in 1922? | National Fascist Party | 1 |
| 4 | Which party became in charge of the Italian government in 1922? | National Fascist Party | 1 |
| 5 | Which party did take control of the government in Italy in the year 1922? | National Fascist Party | 1 |
| 6 | Which party became in charge of the government in Italy in the year 1922? | National Fascist Party | 1 |
| 7 | Which party became in charge of the Italian government in the year 1922? | National Fascist Party | 1 |
| 8 | Which political party assumed governance of Italy in the year 1922? | National Fascist Party | 1 |
| 9 | Which political faction assumed governmental authority in Italy during 1922? | Fascist Party | 1 |

H_sem (k=10): 3 clusters — c0: “National Fascist Party” · c1: “National Fascist Party (Italian Fascism)” · c2: “Fascist Party”
