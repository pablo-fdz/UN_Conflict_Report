# Security Report - Sudan

**Generated on:** 2025-06-09 16:38:35
**Retriever:** HybridCypher
**Configuration:**
- search_params: {'top_k': 5, 'ranker': 'linear', 'alpha': 0.5}
- graphrag_model: gemini-2.5-flash-preview-05-20

---

## Report Content

# Security Report: Sudan - Last Year in Review and Future Outlook

## Overview

The past year in Sudan has been dominated by the escalating civil war between the Sudanese Armed Forces (SAF) and the Rapid Support Forces (RSF) paramilitary group. The conflict has seen significant shifts in tactics and geography, leading to a severe humanitarian crisis and widespread destruction. Events on the ground indicate a protracted struggle with no immediate resolution in sight.

## Key Events and Their Impact (Last Year)

### 1. Escalation in Port Sudan: A Strategic Shift

*   **Event:** In a significant escalation, Port Sudan, a vital government stronghold and the country's main port, experienced its first-ever drone strikes by the RSF.
*   **Impact:** These attacks targeted critical infrastructure, including multiple oil depots, an air base (Osman Digna Air Base), a cargo warehouse, and other civilian facilities. Reports indicated large fires, scattered explosions, and power outages in parts of the city. This marked a strategic shift, bringing the conflict to a previously safer and economically crucial area. Flights were suspended at Port Sudan International Airport following the attacks.
*   **Sources:**
    *   "Editor's note: We are aware of images circulating on social media of a large fire burning at multiple oil depots in Port Sudan, Sudan. The attack has not yet been reported on by Sudanese media, and the cause of the fires is still unclear. The fires come just one day after a series of RSF paramilitary drone strikes in the city, which were the first of the country's two year civil war." - Owen
    *   "Sudanese military, citing official, reports drone attack on Port Sudan targeted civilian facilities including air base and a cargo warehouse; unverified reports claim power outages in parts of city [corrects location struck]"
    *   "Sudanese media report new drone attack on Port Sudan with no specific location given; air defense reported at work"
    *   KG: `RSF paramilitary - PARTICIPATED_IN() -> drones attack`, `oil depots - IS_WITHIN() -> Port Sudan`, `large fire - HAPPENED_IN() -> oil depots`, `Osman Digna Air Base - IS_WITHIN() -> Port Sudan`, `drones attack - HAPPENED_IN() -> Osman Digna Air Base`, `cargo warehouse - IS_WITHIN() -> civilian facilities`, `Power outage - HAPPENED_IN() -> Port Sudan`, `flights suspended - HAPPENED_IN() -> Port Sudan`.

### 2. Contested Territories: Nahud and El Fasher

*   **Event:** The city of Nahud in West Kordofan became a new flashpoint. Reports circulated of RSF forces purportedly entering the city and appearing in front of local administration headquarters. While the Sudanese army quickly countered these claims, stating defenses remained in place and a military garrison was holding out, the incident highlighted the RSF's expanding reach. Previous reports indicated RSF capture of En Nahud, leading to looting and attacks on public buildings.
*   **Event:** In Darfur, particularly around El Fasher, intense fighting persisted. The conflict saw attacks on critical civilian infrastructure, including hospitals (e.g., Social Security Hospital, MSF's hospital in Old Fangak, El Obeid Hospital), and aid convoys.
*   **Impact:** These clashes resulted in significant civilian casualties, exacerbating the humanitarian crisis. Reports of mass graves and the burning of villages underscore the severity of the violence in Darfur.
*   **Sources:**
    *   "Editor's note: We are aware of a report on Al Arabiya, citing a Sudanese army source, saying that defenses remain in place in the city of Nahud in the south of the country. The report appears to come in response to footage circulating on social media platforms purportedly showing Rapid Support Forces militia in front of local administration headquarters in the city. A similar report from Al Jazeera says that a military garrison in Nahud is holding out against RSF attack." - James
    *   KG: `RSF paramilitary - CONFRONTED_WITH() -> Nahud`, `RSF forces - PARTICIPATED_IN() -> Capture of En Nahud`, `looting - HAPPENED_IN() -> En Nahud`, `attacking public buildings - HAPPENED_IN() -> En Nahud`, `RSF - CONFRONTED_WITH(RSF attacked a hospital.) -> hospital`, `RSF strike on hospital - HAPPENED_IN() -> hospital`, `attack on aid convoy - HAPPENED_IN() -> El Fasher`, `mass graves - IS_WITHIN() -> Shagra area`, `burning of eight villages - HAPPENED_IN() -> Shagra area`.

### 3. Persistent Conflict in the Capital Region

*   **Event:** The capital region of Khartoum and Omdurman remained a primary battleground, with both SAF and RSF engaging in shelling and airstrikes.
*   **Impact:** This led to widespread destruction, persistent power outages, and significant civilian casualties. Infrastructure, including power stations and universities, was severely impacted.
*   **Sources:**
    *   KG: `Power outage - HAPPENED_IN() -> Khartoum`, `Bombing of three power stations - HAPPENED_IN() -> Omdurman`, `clashes - HAPPENED_IN() -> Sudan`, `civil war - HAPPENED_IN() -> Sudan`.

### 4. Deepening Humanitarian Crisis

*   **Event:** The ongoing conflict has severely impacted the civilian population, leading to widespread displacement and health emergencies.
*   **Impact:** Cholera outbreaks were reported in multiple states, including Khartoum and Northern State, with hundreds of cases and dozens of deaths. Attacks on prisons, such as in Obeid, led to inmate releases and further instability. The displacement of millions continues, with ongoing assessments of refugee conditions. Aid convoys faced attacks, hindering humanitarian efforts.
*   **Sources:**
    *   KG: `727 cholera cases - HAPPENED_IN() -> Sudan`, `12 deaths - HAPPENED_IN() -> Sudan`, `health ministry - PARTICIPATED_IN() -> 727 cholera cases`, `health ministry - PARTICIPATED_IN() -> 12 deaths`, `strike on a prison - HAPPENED_IN() -> city of Obeid`, `Inmate Release by RSF - HAPPENED_IN() -> prison`, `evacuations - HAPPENED_IN() -> Sudan`, `assessment of refugee conditions - HAPPENED_IN() -> Sudan`, `attack on aid convoy - HAPPENED_IN() -> Sudan`.

### 5. International Dimensions

*   **Event:** The international community has responded with sanctions, with the US imposing measures on Sudan.
*   **Impact:** Sudan, in turn, accused the UAE of violating international conventions, suggesting external involvement in the conflict, particularly in support of the RSF. This highlights the regional and international complexities of the conflict.
*   **Sources:**
    *   KG: `US - CONFRONTED_WITH(US to impose sanctions on Sudan) -> Sudan`, `Sudan - CONFRONTED_WITH(Sudan accuses UAE of violating international convention) -> UAE`, `Emirati - COOPERATED_WITH() -> RSF`, `UAE ships - COOPERATED_WITH() -> RSF`.

## Forward-Looking Perspective

The security situation in Sudan is likely to remain highly volatile and challenging in the coming year.

*   **Continued Conflict and Geographic Spread:** The RSF's ability to conduct drone strikes on strategic locations like Port Sudan indicates a continued intent to disrupt SAF's logistics and international access. This suggests the conflict will not be confined to traditional battlegrounds and could escalate in previously safer areas, increasing the risk to civilian populations and critical infrastructure nationwide.
*   **Worsening Humanitarian Catastrophe:** With ongoing fighting, attacks on aid, and infrastructure damage, the humanitarian crisis is expected to deepen. Cholera outbreaks and other health emergencies will likely intensify, coupled with severe food insecurity and mass displacement.
*   **Economic Deterioration:** Damage to vital infrastructure, including oil depots and power stations, will further cripple Sudan's already fragile economy. This will impact essential services, trade, and the export of South Sudanese oil, leading to increased hardship for the population.
*   **Regional Instability:** The conflict's spillover potential remains high, affecting neighboring countries through refugee flows and potential proxy involvement. External support for either side could further entrench the conflict.
*   **Protracted Stalemate:** Without a decisive military victory by either side or robust, unified international mediation, a political resolution appears distant. The current trajectory points towards a prolonged conflict, with devastating consequences for Sudan's stability and its people.

---

*Report generated using GraphRAG pipeline at 2025-06-09 16:38:35*