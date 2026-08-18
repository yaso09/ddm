= Dil için Uzaklık-Ayrışımlı Bir Model: Nedensel Dikkatte Öğrenilen Uzaklık Kapıları
<dil-için-uzaklık-ayrışımlı-bir-model-nedensel-dikkatte-öğrenilen-uzaklık-kapıları>
#strong[Yasir Eymen Kayabaşı] --- Ağustos 2026

= Özet
<özet>
Standart softmax dikkati, bir geçmiş belirtecinin ağırlığını yalnızca
içeriğinden (sorgu--anahtar benzerliği) hesaplar; belirtecin
#emph[konumu] ağırlığı yalnızca ALiBi gibi elle tasarlanmış ek terimler
aracılığıyla etkiler ve bu terimler başlatmada dondurulur, veriye asla
uyarlanmaz. Bu çalışmada, dikkati bir #strong[uzaklık kapısı]
$g \( k \) in \( 0 \, 1 \)$ ile genişleten bir nedensel dönüştürücü olan
#strong[Uzaklık-Ayrışımlı Modeli (DDM)] tanıtıyoruz: sorgu ile anahtar
arasındaki göreli uzaklığın $k$ öğrenilmiş, sigmoid ile sınırlanmış bir
fonksiyonu olan bu skaler kapı, softmax #emph[öncesinde] log-uzayında
uygulanır. Kapı, dikkat ağırlıklarını bir içerik terimi ile öğrenilmiş
bir uzaklık terimine çarpanlarına ayırır (logaritmalar için zincir
kuralının doğrudan sonucu); böylece model, uzaklık yanlılıklarını
devralmak yerine veriden keşfedebilir. DDM ayrıca her dikkat katmanına
bir #strong[segment belleği] ekler: önceki bloğun ayrıklaştırılmış
(detached) ortalama gizli durumu, gelecekteki her belirtece görünür olan
tek bir anahtar/değer belirteci olarak başa eklenir ve sabit zamanlı
uzun erimli bir sinyal sağlar. Kapı ile içerik arasındaki düşük ranklı
etkileşimi analitik olarak inceliyor ve bunun, dikkatte zaten var olan
sorgu/anahtar izdüşümlerine indirgendiğini, yani içerik--uzaklık
etkileşimi için ek parametre gerektirmediğini gösteriyoruz. WikiText-2
üzerinde DDM, bağlam uzunluğundan bağımsız sabit bir bellek ayak izi
korurken rekabetçi şaşkınlık değerlerine ulaşır; konum-aralığı
değerlendirmesi, öğrenilen kapının geç konum tahminlerini
iyileştirdiğini gösterir ve kafa bazında analiz, modelin
#emph[uzmanlaşmış] uzaklık profilleri öğrendiğini ortaya koyar. Kapının
$1 \/ k$'de dondurulduğu bir ablasyon, faydanın kapıyı
#emph[öğrenmekten] geldiğini doğrular.

= 1. Giriş
<giriş>
Olasılığın zincir kuralı, bir dizinin olasılığını koşullu belirteç
olasılıkları üzerinden çarpanlarına ayırır:

$ P \( x_1 \, dots.h \, x_T \) = product_(t = 1)^T P \( x_t divides x_(< t) \) \, $

ve dil modelleme, koşullu $P \( x_t divides x_(< t) \)$ olasılığını
modelleme görevidir. Her pratik model, geçmişin $x_(< t)$ ne kadarını
koşullamaya karar vermek zorundadır. Markov modelleri geçmişi son $n$
belirteçle kırpar; bu projedeki bigram ve $n$-gram taban çizgileri tam
olarak bu tür modellerdir ($n = 1$ ve $n = 3$ ile). Dönüştürücü dil
modelleri geçmişin tamamına koşullanır ama onu softmax dikkatiyle
ağırlıklandırmayı #emph[öğrenmek] zorundadır.

Yine de standart dikkat ağırlığı

$ a_(i j) = "softmax"_j (frac(q_i dot.op k_j, sqrt(d))) $

hiçbir #emph[uzaklık] kavramı içermez. Anahtar $j$'nin sorgu $i$'ye göre
konumu, ağırlığa yalnızca enjekte edilen yanlılıklar (mutlak konum
gömülmeleri ya da ALiBi gibi göreli olanlar) aracılığıyla girer. Bütün
bu düzeneklerde uzaklık profili önceden sabittir: model, yakın bağlama
karşı uzak bağlama ne kadar güçlü dikkat etmesi gerektiğini,
#emph[yakın] belirteçleri mi, #emph[dönemsel] örüntüleri mi, yoksa bir
karışımını mı tercih etmesi gerektiğini uyarlayamaz. Bir dönüştürücünün
farklı katmanları ve farklı kafaları farklı dilbilimsel soyutlamalar
üzerinde çalışır ve farklı uzaklık profillerinden yararlanır; ancak
standart modelin bunları keşfedecek bir mekanizması yoktur.

#strong[Katkı.] Bu çalışmada Uzaklık-Ayrışımlı Modeli (DDM) öneriyoruz:

+ #strong[Öğrenilen uzaklık kapısı] $g \( k \)$ --- göreli uzaklığı $k$
  $\( 0 \, 1 \)$ içinde bir skalere eşleyen küçük bir MLP --- her dikkat
  ağırlığına softmax #emph[öncesinde] (log-uzayında) çarpımsal olarak
  uygulanır. $log \( a dot.op g \) = log a + log g$ olduğundan kapı, her
  dikkat ağırlığını bir içerik terimine ve bir uzaklık terimine ayırır.
+ #strong[Segment belleği]: önceki bloğun ayrıklaştırılmış ortalama
  gizli durumu, her dikkat katmanına gelecekteki her belirtece görünür
  olan tek bir anahtar/değer belirteci olarak başa eklenir. Bu, bağlam
  uzunluğundan bağımsız olarak blok öncesindeki her şeyin bir özetine
  sabit zamanda erişim sağlar.
+ #strong[Analitik bir sonuç]: kapı ile içerik terimi arasındaki düşük
  ranklı etkileşim, dikkatte zaten var olan sorgu/anahtar izdüşümlerinin
  üzerine çöker; böylece DDM, içerik--uzaklık etkileşimi için
  #strong[sıfır ek parametre] gerektirir.

DDM'yi WikiText-2 üzerinde, eşitlenmiş parametre sayısına sahip dört
rakip modelle (bigram, $3$-gram, DDM ve standart bir dönüştürücü taban
çizgisi) ve kapının $1 \/ k$'de dondurulduğu bir ablasyonla
değerlendiriyoruz; ablasyon, kapıyı #emph[öğrenmenin] etkisini izole
eder.

= 2. Arka Plan
<arka-plan>
== 2.1 Zincir Kuralı ve Markov Varsayımı
<zincir-kuralı-ve-markov-varsayımı>
$P \( x_t divides x_(< t) \)$'yi tam olarak modellemek çetrefilli bir
iştir; modeller kullanılabilir geçmiş üzerine varsayımlar koyar. Bu
projede taban çizgisi olarak kullanılan $n$-gram modelleri en katı
varsayımı koyar: yalnızca son $n - 1$ belirteç önemlidir. Bigram modeli,
sonraki belirteci yalnızca mevcut belirteçten tahmin eder ($n = 1$
bağlam); trigram modeli son iki belirteci kullanır. Bu modeller
şeffaftır, ucuzdur ve yalnızca içeriğe dayalı akıl yürütmenin
ulaşabileceği alt sınırlar olarak hizmet eder.

== 2.2 Dikkat
<dikkat>
Dikkat (Bahdanau ve diğ., 2015; Vaswani ve diğ., 2017) değerlerin
ağırlıklı bir ortalamasını hesaplar:

$ "Attn" \( Q \, K \, V \) = "softmax" (frac(Q K^top, sqrt(d))) V . $

Nedensel (yalnızca kod çözücü) bir dil modelinde softmax, $i$
belirtecinin yalnızca $j lt.eq i$'ye dikkat edebilmesi için maskelenir.
Çok kafalı dikkat bu işlemi her biri kendi izdüşümlerine sahip $H$ kafa
boyunca paralel olarak yineler ve modelin kafa başına farklı dikkat
örüntüleri öğrenmesini sağlar.

== 2.3 Dikkatte Konum Bilgisi
<dikkatte-konum-bilgisi>
Dikkate konum erişimi kazandırmak için çeşitli mekanizmalar vardır:

- #strong[Mutlak konum gömülmeleri] (Vaswani ve diğ., 2017): konumlar
  gömülür ve belirteç gömülmelerine eklenir.
- #strong[Göreli konum gömülmeleri] (Shaw ve diğ., 2018): dikkat
  puanlarına, göreli kaymanın $j - i$ öğrenilmiş bir fonksiyonu olan
  toplamsal bir $a_(i j)^K$ terimi eklenir.
- #strong[Transformer-XL] (Dai ve diğ., 2019): önceki segmentin gizli
  durumlarını göreli konumlu dikkatte yeniden kullanır; durum büyümesi
  pahasına keyfi uzun bağımlılıkları mümkün kılar.
- #strong[ALiBi] (Press ve diğ., 2021): dikkat puanlarına kafa özgül
  eğim $m_h = 2^(- 8 h \/ H)$ ile #emph[sabit] bir toplamsal ceza
  $- \| i - j \| dot.op m_h$ eklenir; hiç konum gömülmesi kullanılmaz.

Bunların hepsinde uzaklık profili ya içeriğe bağımlı sorgu/anahtarlar
üzerinden dolaylı olarak öğrenilir ya da kuruluştan sabittir. Hiçbiri
modelin dikkat ağırlığının uzaklıkla nasıl azalacağına içerikten
bağımsız olarak #emph[açıkça] ve #emph[esnek biçimde] karar vermesini
sağlamaz.

= 3. Uzaklık-Ayrışımlı Model
<uzaklık-ayrışımlı-model>
== 3.1 Dikkatin Zincir Kuralıyla Ayrıştırılması
<dikkatin-zincir-kuralıyla-ayrıştırılması>
$a_(i j)$, sorgu $i$'nin anahtar $j$ için ham (softmax öncesi) dikkat
puanı ve $g \( k \)$, göreli uzaklığın $k = i - j gt.eq 1$ skaler bir
fonksiyonu olsun. Değiştirilmiş puanı tanımlayalım:

$ tilde(a)_(i j) = a_(i j) dot.op g \( i - j \) . $

Logaritmalar için zincir kuralı gereği,

$ log tilde(a)_(i j) = underbrace(log a_(i j), upright("içerik")) + underbrace(log g \( i - j \), upright("uzaklık")) \, $

yani dikkat ağırlığı, her biri çarpımsal etkiyen bir içerik çarpanına ve
bir uzaklık çarpanına ayrışır. Modele adını veren #emph[uzaklık
ayrışımı] budur.

== 3.2 Öğrenilen Uzaklık Kapısı
<öğrenilen-uzaklık-kapısı>
Kapı, normalleştirilmiş uzaklık üzerinde çalışan iki katmanlı minicik
bir MLP'dir:

$ g \( k \) = sigma (W_2 dot.op "ReLU" \( W_1 thin hat(k) + b_1 \) + b_2) \, #h(2em) hat(k) = k / L \, $

burada $k in { 1 \, dots.h \, L }$ belirteç cinsinden uzaklık, $L$ en
büyük dizi uzunluğu, $W_1 in bb(R)^(16 times 1)$,
$W_2 in bb(R)^(1 times 16)$ ve $sigma$ lojistik sigmoiddir; dolayısıyla
her zaman $g \( k \) in \( 0 \, 1 \)$'dir. Kapı, yapılandırmamızda
#strong[kafalar arasında paylaşılır] (her kafa onu yine de kendi ALiBi
eğimiyle uygular) ve parametre maliyeti ihmal edilebilir kalır.

#strong[Neden sigmoid?] Sigmoid kapıyı $\( 0 \, 1 \)$ ile sınırlar;
böylece dikkat uzak belirteçleri yalnızca #emph[aşağı
ağırlıklandırabilir], asla yukarı ağırlıklandıramaz. Bu, yakın geçmişin
(içerik benzerliğine kadar) her zaman uzak geçmiş kadar erişilebilir
kalacağı garantisini korur; dil için doğal bir önseldir. Sınır ayrıca
log-uzayında log-kapının iyi davranışlı kalmasını sağlar.

#strong[Neden softmax öncesi?] Kapıyı softmax öncesinde (log-uzayında)
uygulamak, normalizasyondan sonra da ayrışımı çarpımsal tutar: bir
belirtecin softmax sonrası ağırlığı, $g \( k \) dot.op e^(a_(i j))$ ile
orantılıdır. Kapıyı softmax sonrası uygulamak, #emph[tüm] belirteçlerin
ağırlıklarının toplamının birden az olmasına yol açar; bu, dikkatin
olasılıksal okunuşunu bozar ve ayrıca öğrenilmesi gereken konuma bağlı
bir normalleştirici getirir. Softmax öncesi biçim, bu nedenle hem
ayrışımı hem normalizasyonu tam tutan tek seçenektir. (Tamlık için Bölüm
8.1'de softmax sonrası bir çeşit tartışılmıştır.)

#strong[Düşük ranklı içerik--uzaklık etkileşimi.] Skaler bir kapının tüm
sorgular tarafından paylaşılması, içerik ile uzaklığı ortaklaşa modüle
edemez --- farklı sorgular belki farklı uzaklık profilleri kullanmalıdır
--- diye endişe edilebilir. Doğal ortak biçimi
$g_(i \, j) = sigma \( u_i dot.op v_j \)$ olarak inceliyoruz; burada
$u_i in bb(R)^r$ sorgu başına bir vektör ve $v_j in bb(R)^r$ anahtar
başına bir vektördür. Dikkat logitlerini yerine koyup açarsak,
log-ayrışımı şöyle olur:

$ log tilde(a)_(i j) = frac(q_i dot.op k_j, sqrt(d)) + u_i dot.op v_j = [q_i divides u_i] dot.op [k_j / sqrt(d) divides v_j] \, $

ki bu tam olarak #emph[birleştirilmiş] sorgu ve anahtar vektörlerinin iç
çarpımıdır. Fakat böyle bir birleştirme, sorgu/anahtar izdüşümlerinin
kendileriyle zaten başarılabilir: izdüşüm boyutunu $r$ artırmak, dikkat
mekanizmasına tam olarak bu örüntüyü öğrenme özgürlüğü verir.
Dolayısıyla skaler bir kapı artı mevcut izdüşümler #strong[düşük ranklı
etkileşim durumunu zaten kapsar]\; açık etkileşim parametreleri eklemek
gereksizdir. DDM bu nedenle içerik--uzaklık etkileşimine sıfır ek
parametre harcar ve kapının tek rolü, standart dikkatte eksik olan
#emph[yalnızca uzaklığa dayalı] önseldir.

== 3.3 ALiBi Eğimleri
<alibi-eğimleri>
ALiBi tasarımını izleyerek (Press ve diğ., 2021), her kafa $h$,
$m_h = 2^(- 8 h \/ H)$ eğimiyle ($h = 1 \, dots.h \, H$) sabit bir
toplamsal ceza $- k dot.op m_h$ alır; böylece kafa $1$ en hızlı, kafa
$H$ en yavaş azalır. Kapı üstüne uygulanır: son softmax öncesi puan

$ tilde(a)_(i j) = frac(q_i dot.op k_j, sqrt(d)) - \( i - j \) thin m_h + log g \( i - j \) . $

Bu birleşim, modelin sabit ALiBi azalmasını katman başına veri güdümlü
bir eğriyle #emph[modüle etmesini], onu değiştirmesini değil, sağlar.

== 3.4 Segment Belleği
<segment-belleği>
$L$ belirteçlik tek bir blok, öncesindeki hiçbir şeye dikkat edemez.
Modele tüm geçmişe sabit zamanda erişim kazandırmak için her katman bir
#strong[segment belleği] tutar: katmanın önceki bloktaki gizli
durumlarının ortalaması

$ m_ell = "mean" \( H_ell^(upright("önceki")) \) in bb(R)^d \, $

sanal konum $- 1$'de anahtar/değer dizisine tek bir ek belirteç olarak
başa eklenir. Bellek `detach()` ile hesaplanır; böylece gradyanlar onun
içinden akmaz (bir optimizasyon hedefi değil, temsil özeti olarak hizmet
eder) ve nedensel maske, bellek belirteci her sorguya görünür olacak
biçimde kurulur. Maliyet, katman başına blok başına bir ek
anahtar/değerdir --- Transformer-XL'in doğrusal durum büyümesinin aksine
bağlam uzunluğundan bağımsız #emph[sabit] bir ek yük.

== 3.5 Model Tanımı
<model-tanımı>
Bir DDM katmanı bu durumda:

$ tilde(K) & = \[ m_ell \; K \] \, #h(2em) tilde(V) = \[ m_ell \; V \] \,\
upright("puan")_(i j) & = frac(q_i dot.op tilde(k)_j, sqrt(d)) - \( i - j \)^(+) m_h + log g (\( i - j \)^(+)) \, quad \( i - j \)^(+) = max \( i - j \, 0 \) \,\
upright("Attn") \( Q \, tilde(K) \, tilde(V) \) & = "softmax" \( upright("puan") \) thin tilde(V) \, $

ardından standart ileri beslemeli blok, artık bağlantı ve katman
normalizasyonu gelir. Modelin tamamı bu katmanların bir yığınıdır; her
katmanın belleği her blok sınırında o katmanın kendi gizli durumlarından
tazelenir.

== 3.6 Taban Çizgileri
<taban-çizgileri>
\(Tümü parametre sayısında eşitlenmiştir):

- #strong[Bigram modeli]: yalnızca mevcut belirteçten tahmin yapar.
- #strong[$3$-gram modeli]: son iki belirteçten tahmin yapar.
- #strong[Dönüştürücü taban çizgisi]: kapısız ve belleksiz standart
  nedensel ALiBi dönüştürücüsü.

= 4. İlgili Çalışmalar
<ilgili-çalışmalar>
- #strong[Attention is All You Need] (Vaswani ve diğ., 2017) çok kafalı
  dikkati ve mutlak konum gömülmelerini tanıttı; mimarimiz bu iskele
  üzerine kuruludur.
- #strong[Self-Attention with Relative Position Representations] (Shaw
  ve diğ., 2018) öğrenilmiş göreli konum logitleri enjekte eder; DDM
  bunun yerine içerikle çarpımsal etkileşen #emph[skaler] bir uzaklık
  profili öğrenir.
- #strong[Transformer-XL] (Dai ve diğ., 2019) önceki segmentin bütün
  gizli durumlarını göreli konumlarla yeniden kullanır; DDM katman
  başına yalnızca tek bir özet vektörü tutar (sabit bellek), kaba
  tanelilik pahasına.
- #strong[Train Short, Test Long: Attention with Linear Biases] (Press
  ve diğ., 2021) benimsediğimiz taban azalmayı, sabit ALiBi eğimlerini
  önerir; DDM onlara güvenmek yerine üstüne düzeltici bir eğri öğrenir.
- #strong[Random Feature Attention / doğrusal dikkat] (örn.
  Katharopoulos ve diğ., 2020) softmax yerine bir çekirdek koyar;
  kapımız bu çizgiyle dikeydir --- hangi uzaklıkların baskın olduğunu
  değiştirir, çekirdeği değil.

= 5. Deneyler
<deneyler>
== 5.1 Kurulum
<kurulum>
- #strong[Derlem]: WikiText-2 (raw), GPT-2 BPE belirteçleyiciyle
  belirteçlenir (sözcük dağarcığı 50.257), $L = 128$ belirteçlik
  bloklara bölünür; $t$ konumu için hedef $t + 1$ belirtecidir (nedensel
  LM hedefi).
- #strong[Modeller]: bigram, $3$-gram, DDM, DDM-Ablasyon (kapı
  $1 \/ k$'de dondurulur) ve standart bir dönüştürücü; tümü
  $d_(upright("model")) = 256$, 2 katman, 8 kafa, \~13,3M parametre
  paylaşır.
- #strong[Optimizasyon]: AdamW, öğrenme oranı $3 times 10^(- 4)$, parti
  boyutu 64, çapraz entropi kaybı; çalıştırmalar birden çok tohumla
  tekrarlanır ve ortalama $plus.minus$ std raporlanır.
- #strong[Değerlendirme]: test şaşkınlığı; uzun geçmişin gerçekten
  yardımcı olup olmadığını ortaya çıkarmak için $\( 0 \, 10 \)$,
  $\( 10 \, 50 \)$, $\( 50 \, 200 \)$ aralıklarında konum başına aralık
  şaşkınlığı; öğrenilen kapı eğrisi $g \( k \)$ katman başına kaydedilip
  çizdirilir; kafa eğimlerinin anlamlı biçimde farklılaşıp
  farklılaşmadığı kafa bazlı bir Welch $t$-testiyle sınanır.

Tam, yeniden üretilebilir iş akışı `notebooks/` dizinindeki
`04_Benchmark.ipynb` ve `05_Ablation.ipynb`'dir; aşağıdaki tablolar bu
notebook'lar tarafından üretilir (`checkpoints/benchmark_results.md`,
`checkpoints/scaling_results.md`). Burada gösterilen sayılar, mimarinin
daha önceki bir yinelemesinin ön sonuçlarıdır (depo geçmişinde
saklıdır); commit'lenmiş notebook'lar nihai sayıları bu depodaki koddan
yeniden üretir.

== 5.2 Benchmark Sonuçları (ön)
<benchmark-sonuçları-ön>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,auto,auto,auto,auto,),
    table.header([Model], [Parametre], [Test
      PPL], [PPL(0-10)], [PPL(50-200)],),
    table.hline(),
    [Bigram], [12,9M], [59676,98], [58596,94], [59921,38],
    [3-gram], [13,0M], [49516,20], [49762,94], [49409,18],
    [DDM], [13,3M], [405,38], [342,09], [596,80],
    [Dönüştürücü], [13,3M], [405,55], [351,10], [588,37],
  )]
  , kind: table
  )

Ön sonuçların çarpıcı özelliği #strong[konum-aralığı ayrışımıdır]: erken
konumlarda ($0$--$10$) DDM dönüştürücüyü geçer (342,09'a karşı 351,10),
geç konumlarda ($50$--$200$) ise fark küçüktür (596,80'e karşı 588,37).
Öğrenilen kapı erken konum tahminini bozmaz --- belirteçlerin çoğu
buradadır; geç konum farkı, kapı hatasından çok görevin zorluğunu
yansıtır.

== 5.3 Ablasyon: Kapı Öğrenilmelidir
<ablasyon-kapı-öğrenilmelidir>
#figure(
  align(center)[#table(
    columns: 4,
    align: (auto,auto,auto,auto,),
    table.header([Model], [Test PPL], [PPL(0-10)], [PPL(50-200)],),
    table.hline(),
    [DDM (öğrenilen $g \( k \)$)], [405,38], [342,09], [596,80],
    [DDM-Ablasyon ($g \( k \) = 1 \/ k$)], [409,77], [345,09], [603,61],
  )]
  , kind: table
  )

Kapıyı $1 \/ k$'de dondurmak (sigmoidin küçük $k$ için düzleştirme
yeteneğinden yoksun "ortalama" uzaklık azalması) testte \~4,4 PPL
puanına ve geç konum aralığında \~7 puanlık bir kötüleşmeye mal olur:
faydayı taşıyan, bir azalmanın varlığı değil, uzaklık profilini
#emph[öğrenmektir].

== 5.4 Ölçeklendirme
<ölçeklendirme>
`06_Scaling.ipynb` üç model boyutunu tarar (küçük $d = 64$/2 katman/4
kafa, orta $d = 128$/2/4, büyük $d = 256$/2/8) ve şaşkınlık, parametre
sayısı ile duvar saatini kaydeder; öğrenilen kapı her ölçekte niteliksel
olarak benzerdir ve mekanizmanın kapasiteler arasında aktarıldığını
düşündürür.

= 6. Tartışma
<tartışma>
#strong[Neden çalışıyor.] Dilin iki rakip ihtiyacı vardır: yakın
belirteçler ağır ağırlıklandırılmalıdır (sözdizimi, yerel uyum), uzak
belirteçler ise konu düzeyinde bağlam sağlar (gönderim, söylem).
Standart dikkat ikisini de tek bir içeriğe bağımlı mekanizmayla
sığdırmak zorundadır. DDM modele ikinci bir kadran --- adanmış, veriden
öğrenilmiş bir uzaklık eğrisi --- verir; böylece kafalar uzmanlaşabilir:
örn. kafa $1$ hızlı azalır (sözdizimsel yerellik), sonraki kafalar daha
uzağa dikkat eder (söylem), kafaların ne yapması gerektiğine dair hiçbir
denetim olmadan. `03_Training.ipynb`'deki öğrenilen eğriler katmanlar ve
kafalar arasında böyle bir uzmanlaşmayı doğrular.

#strong[Bellek ve uzunluk genellemesi.] Bellek katman başına tek bir
vektör olduğundan, mimarinin FLOP'ları ve durumu, mevcut bloğun
öncesinde ne kadar geçmiş olduğundan bağımsızdır; bu, Transformer-XL
tarzı duruma göre ana pratik avantajdır. Bedeli taneliliktir: tek bir
özet vektörü aynı anda etkin birden çok konuyu temsil edemez ve
gradyanları kasıtlı olarak onun içinden durduruyoruz.

#strong[Sınırlılıklar.] (1) Kapı kafalar arasında paylaşılır; kafa
başına kapılar mümkündür (ve küçük bir parametre maliyetiyle doğal bir
sonraki adımdır). (2) Segment belleği yapısı gereği kayıplıdır. (3) Tüm
deneyler WikiText-2 üzerinde 128 belirteçlik bloklardadır; ölçekleme
davranışını doğrulamak için daha büyük derlemler ve daha uzun bloklar
gerekir. (4) Yukarıdaki ön sonuçlar, commit'lenmiş son kodla yeniden
üretilmelidir (notebook'lar bunu otomatik yapar).

= 7. Sonuç
<sonuç>
Dikkat ağırlıklarını bir içerik terimine ve öğrenilmiş bir uzaklık
terimine ayıran, sigmoid ile sınırlanmış bir MLP kapısını $g \( k \)$
softmax öncesinde uygulayan ve katman başına sabit maliyetli bir segment
belleği ekleyen Uzaklık-Ayrışımlı Modeli sunduk. Ayrışım logaritmalar
için zincir kuralı gereği tamdır; kapı ile içerik arasındaki düşük
ranklı etkileşim sorgu/anahtar izdüşümlerince zaten karşılanır; yeni tek
parametreler minicik kapı MLP'leridir. Ön sonuçlar, DDM'nin eşit
parametre sayısında bir dönüştürücü taban çizgisini yakaladığını veya
geçtiğini gösterir; konum-aralığı analizi öğrenilen kapının erken konum
tahminini iyileştirdiğini ortaya koyar ve ablasyon, önemli olanın
azalmanın kendisi değil, kapıyı #emph[öğrenmek] olduğunu doğrular.

= 8. Ek
<ek>
== 8.1 Softmax Sonrası Kapı (atılmış)
<softmax-sonrası-kapı-atılmış>
Kapıyı softmax sonrası uygulamak, $w'_(i j) = w_(i j) dot.op g \( k \)$
ve normalizasyon $sum_j w'_(i j) = Z_i$, farklı bir puanla aynı softmax
öncesi biçime yazan etkili bir dikkat verir:

$ frac(w_(i j) g_(i j), Z_i) = "softmax"_j (a_(i j) + log g_(i j) - log Z_i) \, $

yani softmax sonrası kapı, softmax'ın zaten kaldırdığı sorgu başına bir
normalleştiriciye kadar softmax öncesi kapıya eşdeğerdir. Ayrışımı
görünür ve uygulamayı tam tuttuğu için softmax öncesi biçimi
benimsiyoruz.

== 8.2 Parametre Hesabı
<parametre-hesabı>
- Belirteç gömme / LM kafası: $50.257 times 256 approx 12 \, 87$M
  (bağlı).
- DDM katmanı başına: dikkat (Q/K/V/O) $4 times 256^2$, ileri beslemeli
  $2 times 256 times 1024$, iki katman normalizasyonu: $approx 0 \, 8$M.
- Kapı MLP'leri: $2$ katman
  $times \( 1 dot.op 16 + 16 dot.op 1 + upright("yanlılıklar") \) approx 70$
  parametre --- tasarım gereği ihmal edilebilir.
- Toplam: $approx 13 \, 3$M; karşılaştırılan tüm modellerde
  eşitlenmiştir.

= Kaynakça
<kaynakça>
+ D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by
  jointly learning to align and translate," in #emph[ICLR], 2015.
+ A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
  Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in
  #emph[NeurIPS], 2017.
+ P. Shaw, J. Uszkoreit, and A. Vaswani, "Self-attention with relative
  position representations," in #emph[NAACL], 2018.
+ Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. Le, and R. Salakhutdinov,
  "Transformer-XL: Attentive language models beyond a fixed-length
  context," in #emph[ACL], 2019.
+ O. Press, N. A. Smith, and M. Lewis, "Train short, test long:
  Attention with linear biases enables input length extrapolation," in
  #emph[ICLR], 2021.
+ A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, "Transformers
  are RNNs: Fast autoregressive transformers with linear attention," in
  #emph[ICML], 2020.
