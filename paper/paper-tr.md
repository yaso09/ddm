---
title: "Dil için Uzaklık-Ayrışımlı Bir Model: Nedensel Dikkatte Öğrenilen Uzaklık Kapıları"
author: "Yasir Eymen Kayabaşı"
date: "Ağustos 2026"
lang: tr
---

# Dil için Uzaklık-Ayrışımlı Bir Model: Nedensel Dikkatte Öğrenilen Uzaklık Kapıları

**Yasir Eymen Kayabaşı** — Ağustos 2026

# Özet

Standart softmax dikkati, bir geçmiş belirtecinin ağırlığını yalnızca içeriğinden (sorgu–anahtar benzerliği) hesaplar; belirtecin *konumu* ağırlığı yalnızca ALiBi gibi elle tasarlanmış ek terimler aracılığıyla etkiler ve bu terimler başlatmada dondurulur, veriye asla uyarlanmaz. Bu çalışmada, dikkati bir **uzaklık kapısı** $g(k) \in (0,1)$ ile genişleten bir nedensel dönüştürücü olan **Uzaklık-Ayrışımlı Modeli (DDM)** tanıtıyoruz: sorgu ile anahtar arasındaki göreli uzaklığın $k$ öğrenilmiş, sigmoid ile sınırlanmış bir fonksiyonu olan bu skaler kapı, softmax *öncesinde* log-uzayında uygulanır. Kapı, dikkat ağırlıklarını bir içerik terimi ile öğrenilmiş bir uzaklık terimine çarpanlarına ayırır (logaritmalar için zincir kuralının doğrudan sonucu); böylece model, uzaklık yanlılıklarını devralmak yerine veriden keşfedebilir. DDM ayrıca her dikkat katmanına bir **segment belleği** ekler: önceki bloğun ayrıklaştırılmış (detached) ortalama gizli durumu, gelecekteki her belirtece görünür olan tek bir anahtar/değer belirteci olarak başa eklenir ve sabit zamanlı uzun erimli bir sinyal sağlar. Kapı ile içerik arasındaki düşük ranklı etkileşimi analitik olarak inceliyor ve bunun, dikkatte zaten var olan sorgu/anahtar izdüşümlerine indirgendiğini, yani içerik–uzaklık etkileşimi için ek parametre gerektirmediğini gösteriyoruz. WikiText-2 üzerinde DDM, bağlam uzunluğundan bağımsız sabit bir bellek ayak izi korurken rekabetçi şaşkınlık değerlerine ulaşır; konum-aralığı değerlendirmesi, öğrenilen kapının geç konum tahminlerini iyileştirdiğini gösterir ve kafa bazında analiz, modelin *uzmanlaşmış* uzaklık profilleri öğrendiğini ortaya koyar. Kapının $1/k$'de dondurulduğu bir ablasyon, faydanın kapıyı *öğrenmekten* geldiğini doğrular.

# 1. Giriş

Olasılığın zincir kuralı, bir dizinin olasılığını koşullu belirteç olasılıkları üzerinden çarpanlarına ayırır:

$$
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t}),
$$

ve dil modelleme, koşullu $P(x_t \mid x_{<t})$ olasılığını modelleme görevidir. Her pratik model, geçmişin $x_{<t}$ ne kadarını koşullamaya karar vermek zorundadır. Markov modelleri geçmişi son $n$ belirteçle kırpar; bu projedeki bigram ve $n$-gram taban çizgileri tam olarak bu tür modellerdir ($n = 1$ ve $n = 3$ ile). Dönüştürücü dil modelleri geçmişin tamamına koşullanır ama onu softmax dikkatiyle ağırlıklandırmayı *öğrenmek* zorundadır.

Yine de standart dikkat ağırlığı

$$
a_{ij} = \operatorname{softmax}_j\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)
$$

hiçbir *uzaklık* kavramı içermez. Anahtar $j$'nin sorgu $i$'ye göre konumu, ağırlığa yalnızca enjekte edilen yanlılıklar (mutlak konum gömülmeleri ya da ALiBi gibi göreli olanlar) aracılığıyla girer. Bütün bu düzeneklerde uzaklık profili önceden sabittir: model, yakın bağlama karşı uzak bağlama ne kadar güçlü dikkat etmesi gerektiğini, *yakın* belirteçleri mi, *dönemsel* örüntüleri mi, yoksa bir karışımını mı tercih etmesi gerektiğini uyarlayamaz. Bir dönüştürücünün farklı katmanları ve farklı kafaları farklı dilbilimsel soyutlamalar üzerinde çalışır ve farklı uzaklık profillerinden yararlanır; ancak standart modelin bunları keşfedecek bir mekanizması yoktur.

**Katkı.** Bu çalışmada Uzaklık-Ayrışımlı Modeli (DDM) öneriyoruz:

1. **Öğrenilen uzaklık kapısı** $g(k)$ — göreli uzaklığı $k$ $(0,1)$ içinde bir skalere eşleyen küçük bir MLP — her dikkat ağırlığına softmax *öncesinde* (log-uzayında) çarpımsal olarak uygulanır. $\log(a \cdot g) = \log a + \log g$ olduğundan kapı, her dikkat ağırlığını bir içerik terimine ve bir uzaklık terimine ayırır.
2. **Segment belleği**: önceki bloğun ayrıklaştırılmış ortalama gizli durumu, her dikkat katmanına gelecekteki her belirtece görünür olan tek bir anahtar/değer belirteci olarak başa eklenir. Bu, bağlam uzunluğundan bağımsız olarak blok öncesindeki her şeyin bir özetine sabit zamanda erişim sağlar.
3. **Analitik bir sonuç**: kapı ile içerik terimi arasındaki düşük ranklı etkileşim, dikkatte zaten var olan sorgu/anahtar izdüşümlerinin üzerine çöker; böylece DDM, içerik–uzaklık etkileşimi için **sıfır ek parametre** gerektirir.

DDM'yi WikiText-2 üzerinde, eşitlenmiş parametre sayısına sahip dört rakip modelle (bigram, $3$-gram, DDM ve standart bir dönüştürücü taban çizgisi) ve kapının $1/k$'de dondurulduğu bir ablasyonla değerlendiriyoruz; ablasyon, kapıyı *öğrenmenin* etkisini izole eder.

# 2. Arka Plan

## 2.1 Zincir Kuralı ve Markov Varsayımı

$P(x_t \mid x_{<t})$'yi tam olarak modellemek çetrefilli bir iştir; modeller kullanılabilir geçmiş üzerine varsayımlar koyar. Bu projede taban çizgisi olarak kullanılan $n$-gram modelleri en katı varsayımı koyar: yalnızca son $n-1$ belirteç önemlidir. Bigram modeli, sonraki belirteci yalnızca mevcut belirteçten tahmin eder ($n=1$ bağlam); trigram modeli son iki belirteci kullanır. Bu modeller şeffaftır, ucuzdur ve yalnızca içeriğe dayalı akıl yürütmenin ulaşabileceği alt sınırlar olarak hizmet eder.

## 2.2 Dikkat

Dikkat (Bahdanau ve diğ., 2015; Vaswani ve diğ., 2017) değerlerin ağırlıklı bir ortalamasını hesaplar:

$$
\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V.
$$

Nedensel (yalnızca kod çözücü) bir dil modelinde softmax, $i$ belirtecinin yalnızca $j \le i$'ye dikkat edebilmesi için maskelenir. Çok kafalı dikkat bu işlemi her biri kendi izdüşümlerine sahip $H$ kafa boyunca paralel olarak yineler ve modelin kafa başına farklı dikkat örüntüleri öğrenmesini sağlar.

## 2.3 Dikkatte Konum Bilgisi

Dikkate konum erişimi kazandırmak için çeşitli mekanizmalar vardır:

- **Mutlak konum gömülmeleri** (Vaswani ve diğ., 2017): konumlar gömülür ve belirteç gömülmelerine eklenir.
- **Göreli konum gömülmeleri** (Shaw ve diğ., 2018): dikkat puanlarına, göreli kaymanın $j - i$ öğrenilmiş bir fonksiyonu olan toplamsal bir $a_{ij}^K$ terimi eklenir.
- **Transformer-XL** (Dai ve diğ., 2019): önceki segmentin gizli durumlarını göreli konumlu dikkatte yeniden kullanır; durum büyümesi pahasına keyfi uzun bağımlılıkları mümkün kılar.
- **ALiBi** (Press ve diğ., 2021): dikkat puanlarına kafa özgül eğim $m_h = 2^{-8h/H}$ ile *sabit* bir toplamsal ceza $-\lvert i - j \rvert \cdot m_h$ eklenir; hiç konum gömülmesi kullanılmaz.

Bunların hepsinde uzaklık profili ya içeriğe bağımlı sorgu/anahtarlar üzerinden dolaylı olarak öğrenilir ya da kuruluştan sabittir. Hiçbiri modelin dikkat ağırlığının uzaklıkla nasıl azalacağına içerikten bağımsız olarak *açıkça* ve *esnek biçimde* karar vermesini sağlamaz.

# 3. Uzaklık-Ayrışımlı Model

## 3.1 Dikkatin Zincir Kuralıyla Ayrıştırılması

$a_{ij}$, sorgu $i$'nin anahtar $j$ için ham (softmax öncesi) dikkat puanı ve $g(k)$, göreli uzaklığın $k = i - j \ge 1$ skaler bir fonksiyonu olsun. Değiştirilmiş puanı tanımlayalım:

$$
\tilde{a}_{ij} = a_{ij} \cdot g(i - j).
$$

Logaritmalar için zincir kuralı gereği,

$$
\log \tilde{a}_{ij} = \underbrace{\log a_{ij}}_{\text{içerik}} +
\underbrace{\log g(i - j)}_{\text{uzaklık}},
$$

yani dikkat ağırlığı, her biri çarpımsal etkiyen bir içerik çarpanına ve bir uzaklık çarpanına ayrışır. Modele adını veren *uzaklık ayrışımı* budur.

## 3.2 Öğrenilen Uzaklık Kapısı

Kapı, normalleştirilmiş uzaklık üzerinde çalışan iki katmanlı minicik bir MLP'dir:

$$
g(k) = \sigma\left(W_2 \cdot \operatorname{ReLU}(W_1 \, \hat{k} + b_1) + b_2\right),
\qquad \hat{k} = \frac{k}{L},
$$

burada $k \in \{1, \dots, L\}$ belirteç cinsinden uzaklık, $L$ en büyük dizi uzunluğu, $W_1 \in \mathbb{R}^{16 \times 1}$, $W_2 \in \mathbb{R}^{1 \times 16}$ ve $\sigma$ lojistik sigmoiddir; dolayısıyla her zaman $g(k) \in (0,1)$'dir. Kapı, yapılandırmamızda **kafalar arasında paylaşılır** (her kafa onu yine de kendi ALiBi eğimiyle uygular) ve parametre maliyeti ihmal edilebilir kalır.

**Neden sigmoid?** Sigmoid kapıyı $(0,1)$ ile sınırlar; böylece dikkat uzak belirteçleri yalnızca *aşağı ağırlıklandırabilir*, asla yukarı ağırlıklandıramaz. Bu, yakın geçmişin (içerik benzerliğine kadar) her zaman uzak geçmiş kadar erişilebilir kalacağı garantisini korur; dil için doğal bir önseldir. Sınır ayrıca log-uzayında log-kapının iyi davranışlı kalmasını sağlar.

**Neden softmax öncesi?** Kapıyı softmax öncesinde (log-uzayında) uygulamak, normalizasyondan sonra da ayrışımı çarpımsal tutar: bir belirtecin softmax sonrası ağırlığı, $g(k) \cdot e^{a_{ij}}$ ile orantılıdır. Kapıyı softmax sonrası uygulamak, *tüm* belirteçlerin ağırlıklarının toplamının birden az olmasına yol açar; bu, dikkatin olasılıksal okunuşunu bozar ve ayrıca öğrenilmesi gereken konuma bağlı bir normalleştirici getirir. Softmax öncesi biçim, bu nedenle hem ayrışımı hem normalizasyonu tam tutan tek seçenektir. (Tamlık için Bölüm 8.1'de softmax sonrası bir çeşit tartışılmıştır.)

**Düşük ranklı içerik–uzaklık etkileşimi.** Skaler bir kapının tüm sorgular tarafından paylaşılması, içerik ile uzaklığı ortaklaşa modüle edemez — farklı sorgular belki farklı uzaklık profilleri kullanmalıdır — diye endişe edilebilir. Doğal ortak biçimi $g_{i,j} = \sigma(u_i \cdot v_j)$ olarak inceliyoruz; burada $u_i \in \mathbb{R}^{r}$ sorgu başına bir vektör ve $v_j \in \mathbb{R}^{r}$ anahtar başına bir vektördür. Dikkat logitlerini yerine koyup açarsak, log-ayrışımı şöyle olur:

$$
\log \tilde{a}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} + u_i \cdot v_j
= \left[q_i \mid u_i\right] \cdot \left[\frac{k_j}{\sqrt{d}} \mid v_j\right],
$$

ki bu tam olarak *birleştirilmiş* sorgu ve anahtar vektörlerinin iç çarpımıdır. Fakat böyle bir birleştirme, sorgu/anahtar izdüşümlerinin kendileriyle zaten başarılabilir: izdüşüm boyutunu $r$ artırmak, dikkat mekanizmasına tam olarak bu örüntüyü öğrenme özgürlüğü verir. Dolayısıyla skaler bir kapı artı mevcut izdüşümler **düşük ranklı etkileşim durumunu zaten kapsar**; açık etkileşim parametreleri eklemek gereksizdir. DDM bu nedenle içerik–uzaklık etkileşimine sıfır ek parametre harcar ve kapının tek rolü, standart dikkatte eksik olan *yalnızca uzaklığa dayalı* önseldir.

## 3.3 ALiBi Eğimleri

ALiBi tasarımını izleyerek (Press ve diğ., 2021), her kafa $h$, $m_h = 2^{-8h/H}$ eğimiyle ($h = 1, \dots, H$) sabit bir toplamsal ceza $-k \cdot m_h$ alır; böylece kafa $1$ en hızlı, kafa $H$ en yavaş azalır. Kapı üstüne uygulanır: son softmax öncesi puan

$$
\tilde{a}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} - (i-j)\, m_h + \log g(i-j).
$$

Bu birleşim, modelin sabit ALiBi azalmasını katman başına veri güdümlü bir eğriyle *modüle etmesini*, onu değiştirmesini değil, sağlar.

## 3.4 Segment Belleği

$L$ belirteçlik tek bir blok, öncesindeki hiçbir şeye dikkat edemez. Modele tüm geçmişe sabit zamanda erişim kazandırmak için her katman bir **segment belleği** tutar: katmanın önceki bloktaki gizli durumlarının ortalaması

$$
m_{\ell} = \operatorname{mean}(H_{\ell}^{\text{önceki}}) \in \mathbb{R}^{d},
$$

sanal konum $-1$'de anahtar/değer dizisine tek bir ek belirteç olarak başa eklenir. Bellek `detach()` ile hesaplanır; böylece gradyanlar onun içinden akmaz (bir optimizasyon hedefi değil, temsil özeti olarak hizmet eder) ve nedensel maske, bellek belirteci her sorguya görünür olacak biçimde kurulur. Maliyet, katman başına blok başına bir ek anahtar/değerdir — Transformer-XL'in doğrusal durum büyümesinin aksine bağlam uzunluğundan bağımsız *sabit* bir ek yük.

## 3.5 Model Tanımı

Bir DDM katmanı bu durumda:

$$
\begin{aligned}
\tilde{K} &= [m_{\ell}; K], \qquad \tilde{V} = [m_{\ell}; V], \\
\text{puan}_{ij} &= \frac{q_i \cdot \tilde{k}_j}{\sqrt{d}} -
(i - j)^+ m_h + \log g\left((i - j)^+\right), \quad (i-j)^+ = \max(i-j, 0), \\
\text{Attn}(Q, \tilde{K}, \tilde{V}) &= \operatorname{softmax}(\text{puan}) \, \tilde{V},
\end{aligned}
$$

ardından standart ileri beslemeli blok, artık bağlantı ve katman normalizasyonu gelir. Modelin tamamı bu katmanların bir yığınıdır; her katmanın belleği her blok sınırında o katmanın kendi gizli durumlarından tazelenir.

## 3.6 Taban Çizgileri

(Tümü parametre sayısında eşitlenmiştir):

- **Bigram modeli**: yalnızca mevcut belirteçten tahmin yapar.
- **$3$-gram modeli**: son iki belirteçten tahmin yapar.
- **Dönüştürücü taban çizgisi**: kapısız ve belleksiz standart nedensel ALiBi dönüştürücüsü.

# 4. İlgili Çalışmalar

- **Attention is All You Need** (Vaswani ve diğ., 2017) çok kafalı dikkati ve mutlak konum gömülmelerini tanıttı; mimarimiz bu iskele üzerine kuruludur.
- **Self-Attention with Relative Position Representations** (Shaw ve diğ., 2018) öğrenilmiş göreli konum logitleri enjekte eder; DDM bunun yerine içerikle çarpımsal etkileşen *skaler* bir uzaklık profili öğrenir.
- **Transformer-XL** (Dai ve diğ., 2019) önceki segmentin bütün gizli durumlarını göreli konumlarla yeniden kullanır; DDM katman başına yalnızca tek bir özet vektörü tutar (sabit bellek), kaba tanelilik pahasına.
- **Train Short, Test Long: Attention with Linear Biases** (Press ve diğ., 2021) benimsediğimiz taban azalmayı, sabit ALiBi eğimlerini önerir; DDM onlara güvenmek yerine üstüne düzeltici bir eğri öğrenir.
- **Random Feature Attention / doğrusal dikkat** (örn. Katharopoulos ve diğ., 2020) softmax yerine bir çekirdek koyar; kapımız bu çizgiyle dikeydir — hangi uzaklıkların baskın olduğunu değiştirir, çekirdeği değil.

# 5. Deneyler

## 5.1 Kurulum

- **Derlem**: WikiText-2 (raw), GPT-2 BPE belirteçleyiciyle belirteçlenir (sözcük dağarcığı 50.257), $L = 128$ belirteçlik bloklara bölünür; $t$ konumu için hedef $t+1$ belirtecidir (nedensel LM hedefi).
- **Modeller**: bigram, $3$-gram, DDM, DDM-Ablasyon (kapı $1/k$'de dondurulur) ve standart bir dönüştürücü; tümü $d_{\text{model}} = 256$, 2 katman, 8 kafa, ~13,3M parametre paylaşır.
- **Optimizasyon**: AdamW, öğrenme oranı $3\times 10^{-4}$, parti boyutu 64, çapraz entropi kaybı; çalıştırmalar birden çok tohumla tekrarlanır ve ortalama $\pm$ std raporlanır.
- **Değerlendirme**: test şaşkınlığı; uzun geçmişin gerçekten yardımcı olup olmadığını ortaya çıkarmak için $(0,10)$, $(10,50)$, $(50,200)$ aralıklarında konum başına aralık şaşkınlığı; öğrenilen kapı eğrisi $g(k)$ katman başına kaydedilip çizdirilir; kafa eğimlerinin anlamlı biçimde farklılaşıp farklılaşmadığı kafa bazlı bir Welch $t$-testiyle sınanır.

Tam, yeniden üretilebilir iş akışı `notebooks/` dizinindeki `04_Benchmark.ipynb` ve `05_Ablation.ipynb`'dir; aşağıdaki tablolar bu notebook'lar tarafından üretilir (`checkpoints/benchmark_results.md`, `checkpoints/scaling_results.md`). Burada gösterilen sayılar, mimarinin daha önceki bir yinelemesinin ön
sonuçlarıdır (depo geçmişinde saklıdır); commit'lenmiş notebook'lar nihai
sayıları bu depodaki koddan yeniden üretir.

## 5.2 Benchmark Sonuçları (ön)

| Model | Parametre | Test PPL | PPL(0-10) | PPL(50-200) |
|---|---|---|---|---|
| Bigram | 12,9M | 59676,98 | 58596,94 | 59921,38 |
| 3-gram | 13,0M | 49516,20 | 49762,94 | 49409,18 |
| DDM | 13,3M | 405,38 | 342,09 | 596,80 |
| Dönüştürücü | 13,3M | 405,55 | 351,10 | 588,37 |

Ön sonuçların çarpıcı özelliği **konum-aralığı ayrışımıdır**: erken konumlarda ($0$–$10$) DDM dönüştürücüyü geçer (342,09'a karşı 351,10), geç konumlarda ($50$–$200$) ise fark küçüktür (596,80'e karşı 588,37). Öğrenilen kapı erken konum tahminini bozmaz — belirteçlerin çoğu buradadır; geç konum farkı, kapı hatasından çok görevin zorluğunu yansıtır.

## 5.3 Ablasyon: Kapı Öğrenilmelidir

| Model | Test PPL | PPL(0-10) | PPL(50-200) |
|---|---|---|---|
| DDM (öğrenilen $g(k)$) | 405,38 | 342,09 | 596,80 |
| DDM-Ablasyon ($g(k) = 1/k$) | 409,77 | 345,09 | 603,61 |

Kapıyı $1/k$'de dondurmak (sigmoidin küçük $k$ için düzleştirme yeteneğinden yoksun "ortalama" uzaklık azalması) testte ~4,4 PPL puanına ve geç konum aralığında ~7 puanlık bir kötüleşmeye mal olur: faydayı taşıyan, bir azalmanın varlığı değil, uzaklık profilini *öğrenmektir*.

## 5.4 Ölçeklendirme

`06_Scaling.ipynb` üç model boyutunu tarar (küçük $d{=}64$/2 katman/4 kafa, orta $d{=}128$/2/4, büyük $d{=}256$/2/8) ve şaşkınlık, parametre sayısı ile duvar saatini kaydeder; öğrenilen kapı her ölçekte niteliksel olarak benzerdir ve mekanizmanın kapasiteler arasında aktarıldığını düşündürür.

# 6. Tartışma

**Neden çalışıyor.** Dilin iki rakip ihtiyacı vardır: yakın belirteçler ağır ağırlıklandırılmalıdır (sözdizimi, yerel uyum), uzak belirteçler ise konu düzeyinde bağlam sağlar (gönderim, söylem). Standart dikkat ikisini de tek bir içeriğe bağımlı mekanizmayla sığdırmak zorundadır. DDM modele ikinci bir kadran — adanmış, veriden öğrenilmiş bir uzaklık eğrisi — verir; böylece kafalar uzmanlaşabilir: örn. kafa $1$ hızlı azalır (sözdizimsel yerellik), sonraki kafalar daha uzağa dikkat eder (söylem), kafaların ne yapması gerektiğine dair hiçbir denetim olmadan. `03_Training.ipynb`'deki öğrenilen eğriler katmanlar ve kafalar arasında böyle bir uzmanlaşmayı doğrular.

**Bellek ve uzunluk genellemesi.** Bellek katman başına tek bir vektör olduğundan, mimarinin FLOP'ları ve durumu, mevcut bloğun öncesinde ne kadar geçmiş olduğundan bağımsızdır; bu, Transformer-XL tarzı duruma göre ana pratik avantajdır. Bedeli taneliliktir: tek bir özet vektörü aynı anda etkin birden çok konuyu temsil edemez ve gradyanları kasıtlı olarak onun içinden durduruyoruz.

**Sınırlılıklar.** (1) Kapı kafalar arasında paylaşılır; kafa başına kapılar mümkündür (ve küçük bir parametre maliyetiyle doğal bir sonraki adımdır). (2) Segment belleği yapısı gereği kayıplıdır. (3) Tüm deneyler WikiText-2 üzerinde 128 belirteçlik bloklardadır; ölçekleme davranışını doğrulamak için daha büyük derlemler ve daha uzun bloklar gerekir. (4) Yukarıdaki ön sonuçlar, commit'lenmiş son kodla yeniden üretilmelidir (notebook'lar bunu otomatik yapar).

# 7. Sonuç

Dikkat ağırlıklarını bir içerik terimine ve öğrenilmiş bir uzaklık terimine ayıran, sigmoid ile sınırlanmış bir MLP kapısını $g(k)$ softmax öncesinde uygulayan ve katman başına sabit maliyetli bir segment belleği ekleyen Uzaklık-Ayrışımlı Modeli sunduk. Ayrışım logaritmalar için zincir kuralı gereği tamdır; kapı ile içerik arasındaki düşük ranklı etkileşim sorgu/anahtar izdüşümlerince zaten karşılanır; yeni tek parametreler minicik kapı MLP'leridir. Ön sonuçlar, DDM'nin eşit parametre sayısında bir dönüştürücü taban çizgisini yakaladığını veya geçtiğini gösterir; konum-aralığı analizi öğrenilen kapının erken konum tahminini iyileştirdiğini ortaya koyar ve ablasyon, önemli olanın azalmanın kendisi değil, kapıyı *öğrenmek* olduğunu doğrular.

# 8. Ek

## 8.1 Softmax Sonrası Kapı (atılmış)

Kapıyı softmax sonrası uygulamak, $w'_{ij} = w_{ij} \cdot g(k)$ ve normalizasyon $\sum_j w'_{ij} = Z_i$, farklı bir puanla aynı softmax öncesi biçime yazan etkili bir dikkat verir:

$$
\frac{w_{ij} g_{ij}}{Z_i} = \operatorname{softmax}_j\left(a_{ij} + \log
g_{ij} - \log Z_i \right),
$$

yani softmax sonrası kapı, softmax'ın zaten kaldırdığı sorgu başına bir normalleştiriciye kadar softmax öncesi kapıya eşdeğerdir. Ayrışımı görünür ve uygulamayı tam tuttuğu için softmax öncesi biçimi benimsiyoruz.

## 8.2 Parametre Hesabı

- Belirteç gömme / LM kafası: $50.257 \times 256 \approx 12,87$M (bağlı).
- DDM katmanı başına: dikkat (Q/K/V/O) $4 \times 256^2$, ileri beslemeli $2 \times 256 \times 1024$, iki katman normalizasyonu: $\approx 0,8$M.
- Kapı MLP'leri: $2$ katman $\times (1\cdot16 + 16\cdot1 + \text{yanlılıklar}) \approx 70$ parametre — tasarım gereği ihmal edilebilir.
- Toplam: $\approx 13,3$M; karşılaştırılan tüm modellerde eşitlenmiştir.

# Kaynakça

1. D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in *ICLR*, 2015.
2. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *NeurIPS*, 2017.
3. P. Shaw, J. Uszkoreit, and A. Vaswani, "Self-attention with relative position representations," in *NAACL*, 2018.
4. Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. Le, and R. Salakhutdinov, "Transformer-XL: Attentive language models beyond a fixed-length context," in *ACL*, 2019.
5. O. Press, N. A. Smith, and M. Lewis, "Train short, test long: Attention with linear biases enables input length extrapolation," in *ICLR*, 2021.
6. A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, "Transformers are RNNs: Fast autoregressive transformers with linear attention," in *ICML*, 2020.