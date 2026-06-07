# 🎓 COMPLETE PROJECT GUIDE: Highway-Env Reinforcement Learning
## A Mentor's Deep Dive — From Zero to Full Understanding

---

# TABLE OF CONTENTS

1. [Big Picture — What Is This Project?](#1-big-picture)
2. [Reinforcement Learning Fundamentals](#2-rl-fundamentals)
3. [Project Architecture](#3-architecture)
4. [Dependency Deep Dive](#4-dependencies)
5. [File-by-File Breakdown](#5-file-breakdown)
   - 5.1 config.py
   - 5.2 utils.py
   - 5.3 model.py
   - 5.4 train.py
   - 5.5 evaluate.py
6. [Data Flow — Start to Finish](#6-data-flow)
7. [Execution Flow](#7-execution-flow)
8. [Why DQN? The PPO → DQN Story](#8-ppo-to-dqn)
9. [Bugs, Errors, and How They Were Fixed](#9-bugs)
10. [What I Would Improve in V2](#10-v2)
11. [Learning Roadmap](#11-roadmap)

---

# 1. BIG PICTURE — WHAT IS THIS PROJECT? <a name="1-big-picture"></a>

## Beginner Explanation (Gerçek Hayat Analojisi)

Bir araba kullanmayı düşün. İlk defa direksiyona oturduğunda ne yapacağını bilmezsin — rastgele
pedallara basarsın, direksiyonu rastgele çevirirsin. Ama her hata yaptığında (çarptığında,
şeritten çıktığında) "bu kötüydü" dersin, her iyi şey yaptığında (düz gittiğinde, hızlandığında)
"bu iyiydi" dersin. Binlerce deneme sonra araba kullanmayı öğrenirsin.

İşte bu projede bir bilgisayar programı (ajan) tam olarak bunu yapıyor:
- Simüle edilmiş bir otoyolda araba kullanıyor
- Her adımda bir karar veriyor (hızlan, yavaşla, şerit değiştir)
- İyi kararlardan puan alıyor, kötü kararlardan ceza alıyor
- Binlerce deneme yaparak "en iyi strateji nedir" sorusunun cevabını buluyor

## Technical Explanation

Bu bir **value-based off-policy reinforcement learning** projesidir. Bir DQN (Deep Q-Network)
ajanı, highway-env simülasyonunda 5 discrete action arasından seçim yaparak otoyol trafiğinde
güvenli ve hızlı sürüş öğrenir. Ajan, 5x5 kinematics observation'ı input olarak alır,
256x256 fully-connected neural network üzerinden her action için Q-value hesaplar, ve
epsilon-greedy policy ile explore/exploit dengesi kurar.

---

# 2. REINFORCEMENT LEARNING FUNDAMENTALS <a name="2-rl-fundamentals"></a>

## RL'nin 5 Temel Bileşeni

```
┌─────────────────────────────────────────────────────────┐
│                    RL DÖNGÜSÜ                           │
│                                                         │
│   ┌──────────┐    action(a)    ┌─────────────────┐      │
│   │          │ ──────────────► │                 │      │
│   │   AJAN   │                 │    ORTAM        │      │
│   │  (Agent) │ ◄────────────── │  (Environment)  │      │
│   │          │  state(s),      │                 │      │
│   └──────────┘  reward(r)      └─────────────────┘      │
│                                                         │
│   Her adımda:                                           │
│   1. Ajan ortamın durumunu (state) gözlemler            │
│   2. Bir aksiyon (action) seçer                         │
│   3. Ortam yeni duruma geçer ve ödül (reward) verir     │
│   4. Ajan bu deneyimden öğrenir                         │
└─────────────────────────────────────────────────────────┘
```

### Bu projede bunlar ne?

**Agent (Ajan):** Bizim yeşil araba. İçinde bir sinir ağı (neural network) var. Bu ağ,
gördüğü duruma bakıp "hangi aksiyonu yapmalıyım?" sorusuna cevap veriyor.

**Environment (Ortam):** Highway-env simülasyonu. 4 şeritli bir otoyol, 15 tane başka araç,
ve fizik motoru. Ajanın aksiyonunu alıp "dünyayı bir adım ileri götürüyor".

**State (Durum):** Ajanın gördüğü şey. 5x5'lik bir matris:

```
Satır 0: [1.0, 0.88, 0.00, 0.31, 0.00]  ← BİZİM ARAÇ
Satır 1: [1.0, 0.10, 0.75, -0.01, 0.00]  ← En yakın araç 1
Satır 2: [1.0, 0.22, 0.50, -0.01, 0.00]  ← En yakın araç 2
Satır 3: [1.0, 0.33, 0.50, -0.03, 0.00]  ← En yakın araç 3
Satır 4: [0.0, 0.00, 0.00, 0.00, 0.00]  ← Boş slot (araç yok)

Sütunlar: [varlık, x_pozisyon, y_pozisyon, x_hız, y_hız]
```

**Action (Aksiyon):** 5 seçenek:
- 0: Sola şerit değiştir
- 1: Düz devam et
- 2: Sağa şerit değiştir
- 3: Hızlan
- 4: Yavaşla

**Reward (Ödül):** Her adımda bir sayı:
- Hızlı gidersen: +puan (max 0.4)
- Sağ şeritte olursan: +puan (max 0.1)
- Çarparsan: -1.0 (büyük ceza)

## Q-Learning Nedir?

Q-Learning şu soruya cevap arar: "Bu durumda bu aksiyonu yaparsam, toplam ne kadar ödül
kazanırım?"

Bu sorunun cevabına **Q-value** denir:
```
Q(state, action) = "Bu state'te bu action'ı yaparsam, bundan sonra hep en iyi kararları
                    versem toplam ne kadar puan toplarım?"
```

Eğer tüm Q-value'ları bilseydin, en iyi strateji basit olurdu: her zaman en yüksek
Q-value'a sahip aksiyonu seç.

Sorun: state sayısı çok fazla (5x5 matris = sonsuz kombinasyon). Hepsini tablo olarak
tutamazsın. İşte burada **Deep Q-Network** devreye girer.

## DQN = Q-Learning + Neural Network

```
┌─────────────────────────────────────────────────────────┐
│                    DQN MİMARİSİ                         │
│                                                         │
│   State (5x5 = 25 sayı)                                │
│         │                                               │
│         ▼                                               │
│   ┌───────────────┐                                     │
│   │ Linear(25→256) │                                    │
│   │     ReLU       │                                    │
│   └───────┬───────┘                                     │
│           ▼                                             │
│   ┌───────────────┐                                     │
│   │ Linear(256→256)│                                    │
│   │     ReLU       │                                    │
│   └───────┬───────┘                                     │
│           ▼                                             │
│   ┌───────────────┐                                     │
│   │ Linear(256→5)  │  ← 5 output = 5 aksiyon           │
│   └───────┬───────┘                                     │
│           ▼                                             │
│   [Q(s,left), Q(s,idle), Q(s,right), Q(s,fast), Q(s,slow)]
│                                                         │
│   En büyük Q-value'a sahip aksiyonu seç!               │
└─────────────────────────────────────────────────────────┘
```

Neural network, 25 sayıyı (state) alıp 5 sayı (her aksiyonun Q-value'u) üretiyor.
Training sırasında ağın ağırlıkları güncelleniyor ki Q-value tahminleri gerçeğe yaklaşsın.

### DQN'in 3 Kritik Yeniliği

**1. Experience Replay (Deneyim Tekrarı):**
Normal Q-learning'de ajan "gördüğünü hemen öğrenir ve unutur." DQN'de ajan tüm deneyimlerini
bir **replay buffer**'a kaydeder (state, action, reward, next_state). Öğrenirken bu buffer'dan
rastgele örnekler çeker. Bu neden önemli?
- Ardışık deneyimler birbirine çok benzer (korelasyon). Rastgele örnekleme bu korelasyonu kırar.
- Aynı deneyimi birden fazla kez kullanabilirsin (sample efficiency).
- Nadir ama önemli deneyimler (çarpışma gibi) kaybolmaz.

**2. Target Network (Hedef Ağ):**
Ağı güncellerken, güncelleme hedefi de aynı ağdan hesaplanıyor. Bu "hareketli hedef" problemi
yaratır — hedef sürekli kayıyor. DQN çözümü: ikinci bir "donmuş" ağ kopyası tut, hedefleri
ondan hesapla, her 500 adımda bir güncelle.

**3. Epsilon-Greedy Exploration:**
Başlangıçta ajan tamamen rastgele davranır (epsilon=1.0). Zamanla rastgelelik azalır
(epsilon→0.05). Bu, ajanın hem keşfetmesini (exploration) hem de öğrendiklerini kullanmasını
(exploitation) sağlar.

---

# 3. PROJECT ARCHITECTURE <a name="3-architecture"></a>

## Klasör Yapısı

```
highway-rl-project/
│
├── README.md                  ← Proje raporu (GitHub'da görünen)
├── requirements.txt           ← Python kütüphane listesi
├── .gitignore                 ← Git'in yoksayacağı dosyalar
│
├── src/                       ← TÜM KAYNAK KOD
│   ├── config.py              ← Hiperparametreler ve ayarlar
│   │                             (BEYİN: tüm sayısal kararlar burada)
│   │
│   ├── utils.py               ← Yardımcı fonksiyonlar
│   │                             (ortam oluşturma, grafik çizme, callback)
│   │
│   ├── model.py               ← DQN modeli oluşturma ve yükleme
│   │                             (sinir ağı tanımı)
│   │
│   ├── train.py               ← ANA EĞİTİM PROGRAMI
│   │                             (çalıştırılan ilk script)
│   │
│   └── evaluate.py            ← Değerlendirme ve video üretimi
│                                 (çalıştırılan ikinci script)
│
├── assets/                    ← Üretilen görseller
│   ├── evolution.gif           ← Eğitim evrimi videosu
│   └── reward_plot.png         ← Ödül grafiği
│
└── checkpoints/               ← Kaydedilen model ağırlıkları
    ├── model_untrained.zip     ← Eğitim öncesi
    ├── model_halftrained.zip   ← Eğitimin ortası
    └── model_trained.zip       ← Eğitim sonrası
```

## Neden Bu Yapı?

**Separation of Concerns (Kaygıların Ayrılması):**
Her dosyanın TEK bir sorumluluğu var. Bu yazılım mühendisliğinin en temel prensibidir.

- `config.py`: "NE ile?" sorusuna cevap verir (sayılar, ayarlar)
- `model.py`: "NEYİ eğitiyoruz?" sorusuna cevap verir (ağ mimarisi)
- `train.py`: "NASIL eğitiyoruz?" sorusuna cevap verir (eğitim döngüsü)
- `evaluate.py`: "NE KADAR İYİ?" sorusuna cevap verir (test ve video)
- `utils.py`: "ORTAK İŞLER" (herkesçe kullanılan yardımcı fonksiyonlar)

**Config'in ayrı olmasının nedeni:**
Hiperparametre değişikliği için sadece config.py'yi aç, sayıyı değiştir, kaydet.
Hiçbir mantık koduna dokunmana gerek yok. Bunu "hardcoded" yapsaydık, bir learning_rate
değiştirmek için model.py'nin içinde arama yapmak gerekirdi — hata riski yüksek.

**Bu yapı olmasaydı ne olurdu?**
Her şeyi tek dosyaya koysaydın: 500+ satırlık okunması imkansız bir script.
Config ayrı olmasaydı: hiperparametre değiştirmek için kod içinde arama.
Utils ayrı olmasaydı: aynı fonksiyon train.py ve evaluate.py'de tekrar yazılırdı (DRY ihlali).

## Modüller Arası Bağımlılık Grafiği

```
                    config.py
                   ╱    │    ╲
                  ╱     │     ╲
                 ▼      ▼      ▼
           utils.py  model.py  (VideoConfig)
              │ ╲      │         │
              │  ╲     │         │
              ▼   ╲    ▼         ▼
          train.py  └► evaluate.py

Okuma sırası: config → utils → model → train → evaluate
```

Her ok "bağımlıdır" demek. train.py, config'e, utils'e ve model'e bağımlıdır.
Dikkat: evaluate.py, train.py'ye bağımlı DEĞİLDİR — sadece checkpoint dosyalarını okur.

---

# 4. DEPENDENCY DEEP DIVE <a name="4-dependencies"></a>

## requirements.txt Satır Satır

```
gymnasium>=0.29.0      → RL ortam arayüzü standartı
highway-env>=1.8       → Otoyol simülasyonu
stable-baselines3>=2.1.0 → RL algoritma implementasyonları
torch>=2.0.0           → Neural network framework
numpy>=1.24.0          → Sayısal hesaplamalar
matplotlib>=3.7.0      → Grafik çizme
Pillow>=10.0.0         → GIF oluşturma (görüntü işleme)
```

### Neden Gymnasium?

**Ne:** OpenAI Gym'in devamı. RL ortamları için standart bir API tanımlar.

**Neden:** Tüm RL kütüphaneleri (Stable-Baselines3, RLlib, CleanRL) Gymnasium API'sını
kullanır. Bu API şunları standartlaştırır:
- `env.reset()` → ortamı sıfırla, ilk state'i ver
- `env.step(action)` → aksiyon uygula, yeni state + reward + done ver
- `env.render()` → görselleştir

**Olmasaydı:** Her ortam farklı API kullanırdı. Bir algoritmayı farklı ortamlarda test
etmek imkansız olurdu.

### Neden highway-env?

**Ne:** Gymnasium API'sını implement eden bir otoyol simülasyonu. Farama Foundation
tarafından geliştiriliyor.

**Neden bu ortam seçildi (ödeve göre):** Projenin Option A'sında belirtilen ortam.
CPU'da çalışır (GPU gerekmez), hızlı simülasyon, görsel çıktı üretir.

**Olmasaydı:** Ortamı sıfırdan yazmak gerekirdi — fizik motoru, trafik simülasyonu,
render sistemi... Bu başlı başına 6 aylık bir proje.

### Neden Stable-Baselines3 (SB3)?

**Ne:** PyTorch tabanlı, test edilmiş, dokümantasyonu mükemmel RL algoritma kütüphanesi.
DQN, PPO, A2C, SAC, TD3 gibi algoritmaları hazır sunar.

**Neden:** Algoritmaları sıfırdan yazmak yerine kanıtlanmış implementasyonları kullanmak:
- Daha az bug riski
- Callback sistemi (eğitim sırasında özel işlemler)
- Checkpoint kaydetme/yükleme
- Otomatik tensorboard logging

**Alternatif: CleanRL** — Daha minimal, tek dosyalık implementasyonlar. Öğrenmek için
güzel ama production-ready değil. Biz SB3'ü seçtik çünkü callback ve checkpoint sistemi
projenin ihtiyaçlarına uygun.

**Olmasaydı:** DQN'i sıfırdan yazmak gerekirdi — replay buffer, target network, epsilon
decay, gradient descent... Bu 500+ satır ekstra kod ve çok fazla bug potansiyeli demek.

### Neden PyTorch?

**Ne:** Facebook (Meta) tarafından geliştirilen deep learning framework.

**Neden:** SB3 arkada PyTorch kullanıyor. Biz doğrudan PyTorch'a dokunmuyoruz (SB3
soyutluyor) ama `nn.ReLU` gibi activation function tanımları için import ediyoruz.

**Alternatif: TensorFlow** — SB3, TensorFlow'u desteklemiyor (eskiden TF1 versiyonu
vardı, artık sadece PyTorch). Bu yüzden seçim zorunlu.

### Neden NumPy, Matplotlib, Pillow?

- **NumPy:** Her sayısal hesaplamanın temeli. Rolling average, frame arrays, reward lists.
- **Matplotlib:** Reward ve episode length grafikleri çizmek.
- **Pillow:** Matplotlib frame'lerini PIL Image'a çevirip GIF olarak kaydetmek.

---

# 5. FILE-BY-FILE BREAKDOWN <a name="5-file-breakdown"></a>

## 5.1 config.py — Projenin Beyni

### Ne Yapar?
Tüm sayısal kararları (hiperparametreler) ve ortam ayarlarını tek bir yerde toplar.
Hiçbir mantık kodu yoktur — sadece veri tanımları.

### Neden Dataclass?

```python
from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    lanes_count: int = 4
    vehicles_count: int = 15
    ...
```

`@dataclass` ne yapar? Python'a der ki: "Bu sınıfın amacı veri tutmak. Benim için
`__init__`, `__repr__`, `__eq__` metodlarını otomatik oluştur."

**Dataclass olmasaydı:**
```python
# KÖTÜ — her değişiklikte __init__'i de güncelle
class EnvConfig:
    def __init__(self):
        self.lanes_count = 4
        self.vehicles_count = 15
        self.duration = 40
        # ... 15 satır daha
```

**Neden dict değil?**
```python
# KÖTÜ — typo fark edilmez, IDE otomatik tamamlama çalışmaz
config = {"lanes_count": 4, "vehicels_count": 15}  # "vehicels" typo!
config["vehicels_count"]  # Hata vermez, None döner
```

Dataclass ile `config.vehicels_count` yazmaya çalışsan IDE kırmızı çizer.

### field(default_factory=lambda: [...]) Neden?

```python
observation_features: List[str] = field(
    default_factory=lambda: ["presence", "x", "y", "vx", "vy"]
)
```

**Problem:** Python'da mutable default arguments (list, dict) tehlikelidir:
```python
# YANLIŞ — tüm instance'lar AYNI listeyi paylaşır!
class Bad:
    features: list = ["a", "b"]

a = Bad()
b = Bad()
a.features.append("c")
print(b.features)  # ["a", "b", "c"] — b de etkilendi!
```

`default_factory` her instance için YENİ bir liste oluşturur.

### to_env_config() Metodu

```python
def to_env_config(self) -> Dict[str, Any]:
    return {
        "lanes_count": self.lanes_count,
        "vehicles_count": self.vehicles_count,
        ...
    }
```

**Ne yapar:** Bizim güzel yapılı dataclass'ı, highway-env'in beklediği dict formatına
çevirir.

**Neden gerekli:** highway-env, `gym.make("highway-v0", config={...})` şeklinde bir
dictionary bekler. Bizim dataclass'ı doğrudan veremeyiz.

**Olmasaydı:** Her yerde `{"lanes_count": config.lanes_count, ...}` yazmak gerekirdi.
DRY (Don't Repeat Yourself) ihlali.

### TrainConfig — DQN Hiperparametreleri Satır Satır

```python
total_timesteps: int = 30_000   # Toplam eğitim adımı
```
**Neden 30K?** Daha az = yetersiz öğrenme. Daha fazla = uzun süre + diminishing returns.
30K, bu ortam için "yeterince iyi" bir denge.

```python
learning_rate: float = 5e-4     # 0.0005
```
**Neden 5e-4?** Neural network ağırlıklarının her adımda ne kadar değişeceğini belirler.
- Çok büyük (1e-2): Ağırlıklar çok hızlı değişir → kararsız öğrenme, overshooting
- Çok küçük (1e-6): Ağırlıklar çok yavaş değişir → 30K step yetmez
- 5e-4: DQN için standart başlangıç noktası

```python
buffer_size: int = 50_000       # Replay buffer kapasitesi
```
**Neden 50K?** Son 50K deneyimi hafızada tutar.
- Çok küçük (1K): Ajan sadece son deneyimlerden öğrenir → çeşitlilik az
- Çok büyük (1M): RAM tüketir, eski irrelevant deneyimler kalır
- 50K: Bu ortam için yeterli çeşitlilik sağlar

```python
learning_starts: int = 1_000    # Eğitime başlamadan önce rastgele adım
```
**Neden 1K?** İlk 1000 adım tamamen rastgele. Buffer'ı çeşitli deneyimlerle doldurur.
Boş buffer'dan öğrenmeye başlamak: overfitting riski. 1000 adım ≈ 5-10 episode.

```python
target_update_interval: int = 500
```
**Neden 500?** Target network her 500 adımda güncellenir.
- Çok sık (10): Hareketli hedef problemi çözülmez
- Çok nadir (10K): Target çok geride kalır, öğrenme yavaşlar

```python
exploration_fraction: float = 0.3    # Toplam eğitimin %30'u exploration
exploration_initial_eps: float = 1.0  # Başlangıç: %100 rastgele
exploration_final_eps: float = 0.05   # Son: %5 rastgele
```
**Neden bu değerler?** Epsilon-greedy schedule:
```
Step 0       → epsilon = 1.0 (tamamen rastgele)
Step 9000    → epsilon = 0.05 (neredeyse tamamen greedy)
Step 9001-30K → epsilon = 0.05 (sabit kalır)

%30 x 30000 = 9000 adımda 1.0'dan 0.05'e düşer
```
%5 final exploration = "öğrendim ama yine de ara sıra yeni şeyler dene"

---

## 5.2 utils.py — Ortak Altyapı

### make_env() Fonksiyonu

```python
def make_env(env_config: EnvConfig, seed: int = 42) -> gym.Env:
    env = gym.make(
        env_config.env_id,
        render_mode="rgb_array",
        config=env_config.to_env_config(),
    )
    env.reset(seed=seed)
    env = Monitor(env)
    return env
```

**Satır satır:**

1. `gym.make(...)` — Gymnasium'a "highway-v0 ortamını oluştur" der.
   - `render_mode="rgb_array"`: Ekrana çizmek yerine numpy array olarak ver (video için)
   - `config=...`: Highway ortamının ayarlarını geç

2. `env.reset(seed=seed)` — Ortamı başlangıç durumuna getir.
   - `seed=42`: Rastgeleliği sabitler. Aynı seed = aynı başlangıç pozisyonu.
   - **Neden seed?** Reproducibility (tekrarlanabilirlik). Hoca aynı kodu çalıştırınca
     aynı sonucu almalı.

3. `env = Monitor(env)` — Ortamı bir "izleme zarfına" sarar.
   - Monitor ne yapar: Her episode'un toplam ödülünü ve uzunluğunu otomatik kaydeder.
   - `info["episode"]["r"]` ve `info["episode"]["l"]` verilerini sağlar.
   - **Olmasaydı:** Ödülleri kendimiz saymamız gerekirdi (hata riski).

**Neden fonksiyon?** Ortam oluşturma 3 yerde yapılıyor (train, evaluate, model). Her
seferinde bu 5 satırı tekrar yazmak yerine tek fonksiyon çağrısı.

### RewardLoggerCallback Sınıfı

```python
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True
```

**Ne yapar:** Eğitim sırasında her adımda otomatik çağrılır. Episode bittiğinde toplam
ödülü ve uzunluğu listeye ekler.

**Callback pattern ne?** SB3'ün eğitim döngüsü kapalı bir kutu — `model.learn()` çağırdığında
içeride ne olduğunu göremezsin. Callback, bu kutunun içine "casus" sokmak gibi. Her adımda
`_on_step()` çağrılır.

**`self.locals` ne?** SB3'ün eğitim döngüsünün yerel değişkenleri. `infos` key'i,
ortamın her step'te döndürdüğü ekstra bilgileri içerir.

**`if "episode" in info` neden?** Sadece episode BİTTİĞİNDE `info` dict'inde "episode"
key'i oluşur. Episode devam ederken bu key yok. Bu, Monitor wrapper'ın davranışıdır.

**`return True` neden?** True = eğitime devam et. False dönersen eğitim durur. Bunu
early stopping için kullanabilirsin ama biz kullanmıyoruz.

**Olmasaydı:** Eğitim sırasında hiçbir veri toplamazdık → grafik çizemezdik → proje
notunun %35'i olan "Training Analysis" bölümü boş kalırdı.

### plot_training_results() Fonksiyonu

```python
def plot_training_results(rewards, lengths, save_path, window=20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
```

**Ne yapar:** İki yan yana grafik çizer — reward ve episode length.

**Rolling average neden?**
```python
rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
```
Raw reward verileri çok gürültülü — her episode'da ödül farklı. 20-episode rolling
average (hareketli ortalama) trendi gösterir.

```
np.convolve([10, 20, 30, 40, 50], [0.33, 0.33, 0.33], mode="valid")
→ [20, 30, 40]  ← her 3'lü grubun ortalaması
```

**`mode="valid"` neden?** Sadece tam pencere sığan yerlerde hesapla. İlk 19 episode'da
rolling average yok çünkü 20 veri noktası dolmamış.

**`alpha=0.3` neden?** Raw veriler şeffaf çizilir ki rolling average öne çıksın.

**`dpi=150` neden?** Yüksek çözünürlük. Default 72 dpi README'de bulanık görünür.

### evaluate_agent() Fonksiyonu

```python
def evaluate_agent(model, env, n_episodes=10):
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
```

**`deterministic=True` kritik!** Eğitimde ajan bazen rastgele davranır (exploration).
Değerlendirmede HER ZAMAN en iyi bildiği aksiyonu seçmeli. `deterministic=True` bunu
sağlar — epsilon=0 ile çalışır.

**`terminated` vs `truncated`:** Gymnasium'da iki farklı bitiş var:
- `terminated=True`: Doğal bitiş (çarpışma)
- `truncated=True`: Zaman doldu (duration limiti)
İkisi de episode'u bitirir ama farklı anlamları var.

---

## 5.3 model.py — Sinir Ağı Fabrikası

### ACTIVATION_MAP

```python
ACTIVATION_MAP = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "LeakyReLU": nn.LeakyReLU,
}
```

**Ne yapar:** Config'deki string ismi ("ReLU") gerçek PyTorch sınıfına çevirir.

**Neden string→class mapping?** Config dosyasında `nn.ReLU` yazamazsın (import gerekir).
String olarak tutup burada çevirmek, config'i saf veri olarak tutmamızı sağlar.

**ReLU nedir?** Rectified Linear Unit: `f(x) = max(0, x)`
- Negatif değerleri sıfırlar, pozitif değerleri olduğu gibi geçirir.
- En yaygın activation function. Basit, hızlı, gradient vanishing problemi yok.

**Tanh neden reddedildi?** Tanh çıktısı [-1, 1] arasında. Q-value'lar bu aralığa
sığmayabilir (bizim ödüllerimiz 0-200 arasında). Tanh, output'u sıkıştırırdı.

### create_model() — DQN Oluşturma

```python
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=train_config.learning_rate,
    buffer_size=train_config.buffer_size,
    ...
)
```

**`policy="MlpPolicy"` ne demek?** MLP = Multi-Layer Perceptron = Fully Connected
Neural Network. Alternatif: `CnnPolicy` (görüntü inputları için). Highway-env'de
input 25 sayı olduğu için MLP yeterli.

**Her parametre config'den geliyor.** Model dosyası hiçbir sayısal karar içermiyor.
Bu, Single Responsibility Principle: model.py'nin görevi "oluştur", config.py'nin görevi
"parametreleri belirle".

### load_model() — Checkpoint Yükleme

```python
def load_model(model_path: str, env_config: EnvConfig) -> DQN:
    env = make_env(env_config)
    model = DQN.load(model_path, env=env)
    return model
```

**Neden env gerekiyor?** SB3 modeli yüklerken ortamın observation/action space'ini
bilmesi lazım. Ağırlıklar .zip dosyasından gelir ama ortam bilgisi "canlı" olarak
verilmeli.

---

## 5.4 train.py — Ana Eğitim Pipeline'ı

### Execution Flow

```
train() çağrıldığında:
│
├── 1. Config nesneleri oluştur (EnvConfig, TrainConfig)
│
├── 2. Klasörleri oluştur (checkpoints/, logs/, assets/)
│
├── 3. Model oluştur (create_model)
│     └── İçinde: env oluştur → DQN nesnesi yarat
│
├── 4. UNTRAINED checkpoint kaydet
│     └── Rastgele ağırlıklarla model → model_untrained.zip
│
├── 5. İlk yarı eğitim (15K step)
│     └── model.learn(15000, callback=reward_logger)
│     └── HALFTRAINED checkpoint kaydet
│
├── 6. İkinci yarı eğitim (15K step)
│     └── model.learn(15000, reset_num_timesteps=False)
│     └── TRAINED checkpoint kaydet
│
├── 7. Grafik çiz ve kaydet
│     └── plot_training_results(rewards, lengths)
│
└── 8. Final modeli değerlendir
      └── evaluate_agent(model, env, 5 episode)
```

### Neden Eğitim İkiye Bölünüyor?

```python
half_steps = train_config.total_timesteps // 2

# İlk yarı
model.learn(total_timesteps=half_steps, callback=reward_callback)
model.save("model_halftrained")

# İkinci yarı
model.learn(total_timesteps=half_steps, reset_num_timesteps=False)
model.save("model_trained")
```

**Proje gerekliliği:** Ödev 3 aşamalı video istiyor — untrained, half-trained, trained.
Bu yüzden eğitimi ortada durdurup checkpoint kaydetmemiz gerekiyor.

**`reset_num_timesteps=False` kritik!** Bu olmadan SB3, ikinci `learn()` çağrısında
timestep sayacını sıfırlar. Bu, epsilon decay'i baştan başlatır (tekrar full exploration).
`False` diyerek "kaldığın yerden devam et" diyoruz.

**Olmasaydı:** İkinci yarıda epsilon tekrar 1.0'dan başlar → ajan "unutur" ve tekrar
rastgele davranır → fully trained model, half-trained'den kötü olur (PPO'da yaşadığımız
sorunun benzerini yaratır).

### __name__ == "__main__" Pattern

```python
if __name__ == "__main__":
    train()
```

**Ne yapar:** Bu dosya doğrudan çalıştırıldığında (`python train.py`) train() çalışır.
Başka bir dosyadan import edildiğinde çalışmaz.

**Neden?** evaluate.py'de `from train import something` yapabilmek için. Import
yapıldığında eğitimi otomatik başlatmak istemezsin.

---

## 5.5 evaluate.py — Değerlendirme ve Video Üretimi

### record_episode_frames()

```python
def record_episode_frames(model, env_config, seed=42):
    env = gym.make(env_config.env_id, render_mode="rgb_array", config=...)
    obs, _ = env.reset(seed=seed)
    frames = []
    done = False

    while not done:
        frame = env.render()        # Mevcut durumun görselini al
        if frame is not None:
            frames.append(frame)    # Listeye ekle
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return frames
```

**Ne yapar:** Bir episode boyunca her kareyi (frame) kaydeder. Sonuç: numpy array listesi.

**Neden `render_mode="rgb_array"`?** Ekrana çizmek yerine her kareyi numpy array olarak
döndürür. Bu array'leri birleştirip GIF yapacağız.

**Neden ayrı env oluşturuyor (make_env kullanmıyor)?** Monitor wrapper render'ı
etkileyebilir. Temiz, wrapper'sız bir ortam istiyoruz.

### find_best_seed() — Video Kalitesi İçin Kritik

```python
def find_best_seed(model, env_config, seeds):
    best_seed = seeds[0]
    best_length = 0

    for seed in seeds:
        env = gym.make(...)
        obs, _ = env.reset(seed=seed)
        length = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            length += 1
            done = terminated or truncated
        if length > best_length:
            best_length = length
            best_seed = seed

    return best_seed
```

**Ne yapar:** 13 farklı seed dener, ajanın en uzun hayatta kaldığı seed'i bulur.

**Neden bu gerekli?** Highway-env'de trafik seed'e göre değişir. Bazı seed'lerde
ajanın hemen önüne araç çıkar ve çarpma kaçınılmaz olur. Video için ajanın davranışını
GÖRMEK istiyoruz — 18 frame'lik çarpma videosu hiçbir şey göstermez.

**Bu fonksiyon olmasaydı ne olurdu?** İlk versiyonda yoktu. seed=42 sabit kullanıldı.
Sonuç: üç aşamada da 18 frame — hepsi çarpıyordu. Video anlamsızdı.

**Edge case:** Untrained ajan için bile en iyi seed'i buluyoruz. Rastgele davranış bile
bazı seed'lerde uzun süre hayatta kalabilir — bu "şans eseri düz gitme" davranışını
gösterir ki bu da eğitici bir gözlem.

### create_evolution_gif() — Video Oluşturma

```python
# Frame sayılarını eşitle (kısa olanları son frame ile doldur)
max_frames = max(len(f) for f in all_stage_frames)
for i, frames in enumerate(all_stage_frames):
    while len(frames) < max_frames:
        frames.append(frames[-1])
```

**Neden padding?** Untrained 107 frame, Half-trained 71 frame, Trained 200 frame.
GIF'te yan yana gösteriyoruz — frame sayıları eşit olmalı. Kısa videoların sonuna
"son kare" tekrar ekleniyor (donuk kalıyor).

```python
# Matplotlib → PIL Image dönüşümü
fig.canvas.draw()
img_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
combined_frames.append(Image.fromarray(img_array))
```

**Ne oluyor burada?**
1. `fig.canvas.draw()` — Matplotlib figürünü bellekte çizer
2. `buffer_rgba()` — Çizilen görüntüyü RGBA (4 kanal) numpy array olarak al
3. `[:, :, :3]` — Alpha kanalını at, sadece RGB kalsın (GIF alpha desteklemez)
4. `Image.fromarray()` — Numpy array'i PIL Image'a çevir

**Neden Matplotlib üzerinden?** Doğrudan frame'leri birleştirebilirdik ama Matplotlib
ile başlık (title) ekliyoruz: "Untrained", "Half-Trained", "Fully Trained".

```python
# GIF kaydetme
combined_frames[0].save(
    gif_path,
    save_all=True,
    append_images=combined_frames[1:],
    duration=1000 // video_config.fps,  # ms cinsinden frame süresi
    loop=0,  # 0 = sonsuz döngü
)
```

**Pillow GIF API:** İlk frame `.save()` ile, geri kalanlar `append_images` ile eklenir.
`duration=100` ms (10 FPS) → her kare 0.1 saniye gösterilir.

---

# 6. DATA FLOW — START TO FINISH <a name="6-data-flow"></a>

```
┌─────────────────── TRAINING DATA FLOW ───────────────────┐
│                                                           │
│  config.py                                                │
│  (sayılar) ─────┐                                        │
│                  ▼                                        │
│  utils.py:make_env()                                      │
│  (ortam oluştur) ───┐                                    │
│                      ▼                                    │
│  model.py:create_model()                                  │
│  (DQN + ortam birleştir) ───┐                            │
│                              ▼                            │
│  train.py:train()                                         │
│  ┌──────────────────────────────────────┐                │
│  │  model.learn(15000)                   │                │
│  │  ┌────────────────────────────┐       │                │
│  │  │  Her adımda:               │       │                │
│  │  │  1. env → obs (5x5 matris) │       │                │
│  │  │  2. model → action (0-4)   │       │                │
│  │  │  3. env.step(action)       │       │                │
│  │  │     → new_obs, reward, done│       │                │
│  │  │  4. Buffer'a kaydet        │       │                │
│  │  │  5. Buffer'dan sample al   │       │                │
│  │  │  6. Neural network güncelle│       │                │
│  │  │  7. Callback: episode      │       │                │
│  │  │     bittiyse reward logla  │       │                │
│  │  └────────────────────────────┘       │                │
│  │                                       │                │
│  │  → model_halftrained.zip              │                │
│  │  → model_trained.zip                  │                │
│  │  → reward_plot.png                    │                │
│  └──────────────────────────────────────┘                │
│                                                           │
└───────────────────────────────────────────────────────────┘

┌─────────────────── EVALUATION DATA FLOW ─────────────────┐
│                                                           │
│  checkpoints/*.zip ──► model.py:load_model()              │
│                           │                               │
│                           ▼                               │
│  evaluate.py:find_best_seed()                             │
│  (13 seed dene → en uzun hayatta kalanı seç)              │
│                           │                               │
│                           ▼                               │
│  evaluate.py:record_episode_frames()                      │
│  (seçilen seed ile episode kaydı → frame listesi)         │
│                           │                               │
│                           ▼                               │
│  evaluate.py:create_evolution_gif()                       │
│  (3 stage × N frame → matplotlib → PIL → GIF)            │
│                           │                               │
│                           ▼                               │
│  assets/evolution.gif                                     │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

# 7. EXECUTION FLOW <a name="7-execution-flow"></a>

```
Kullanıcı: python train.py
│
├── Python: train.py yükle
├── Import: config, model, utils modüllerini yükle
├── __name__ == "__main__" → True → train() çağır
│
├── EnvConfig() oluştur ← tüm default değerlerle
├── TrainConfig() oluştur ← tüm default değerlerle
│
├── os.makedirs("checkpoints") ← yoksa oluştur, varsa dokunma
├── os.makedirs("logs")
├── os.makedirs("assets")
│
├── create_model(env_config, train_config) çağır
│   ├── make_env() çağır
│   │   ├── gym.make("highway-v0") → HighwayEnv nesnesi
│   │   ├── env.reset(seed=42) → ilk state
│   │   └── Monitor(env) → sarmalanmış env
│   ├── DQN(...) oluştur → model nesnesi
│   └── return model
│
├── model.save("model_untrained") → .zip dosyası
│   └── İçinde: policy.pth (ağırlıklar) + metadata
│
├── RewardLoggerCallback() oluştur → boş listelerle
│
├── model.learn(15000, callback=reward_logger)
│   ├── [İç döngü - SB3 tarafından yönetilir]
│   ├── İlk 1000 step: tamamen rastgele (learning_starts)
│   ├── Step 1001+: her 4 step'te bir gradient update
│   ├── Her episode bitiminde: callback listlere reward ekler
│   └── 15000 step sonra durur
│
├── model.save("model_halftrained")
│
├── model.learn(15000, reset_num_timesteps=False)
│   ├── Epsilon kaldığı yerden devam eder
│   └── Buffer'daki eski deneyimler korunur
│
├── model.save("model_trained")
│
├── plot_training_results(rewards, lengths, "assets/reward_plot.png")
│   └── matplotlib → .png dosyası
│
├── evaluate_agent(model, eval_env, 5)
│   ├── 5 episode çalıştır (deterministic)
│   ├── Ortalama ve std hesapla
│   └── Yazdır: "Evaluation: 69.97 ± 40.23"
│
└── "Training complete!" yazdır
```

---

# 8. WHY DQN? THE PPO → DQN STORY <a name="8-ppo-to-dqn"></a>

## İlk Deneme: PPO (Proximal Policy Optimization)

### PPO Nasıl Çalışır?
PPO bir **policy gradient** yöntemidir. Q-value öğrenmek yerine, doğrudan "hangi
durumda hangi aksiyonu ne olasılıkla seç" politikasını öğrenir.

PPO'nun özelliği: politika güncellemelerini "kırpma" (clipping) ile sınırlar.
Bu, bir güncellemenin politikayı çok fazla değiştirmesini önler.

### PPO ile Ne Oldu?

```
Eğitim Sonuçları:
  8K steps:  143.43 ± 5.18   ← İYİ! Ama az step
  50K steps: 61.97 ± 55.60   ← KÖTÜ! Yüksek varyans

Video Sonuçları (50K):
  Untrained:    18 frame  (çarpma)
  Half-Trained: 200 frame (hayatta!)
  Fully Trained: 18 frame (çarpma!) ← DAHA KÖTÜ!
```

**Fully trained ajan, half-trained'den KÖTÜ!** Bu "catastrophic forgetting" veya
"policy collapse" denir.

### Neden PPO Başarısız Oldu?

**1. On-policy problem:**
PPO **on-policy** bir algoritmadır — sadece mevcut politikanın ürettiği veriden öğrenir.
Eski deneyimleri ATAR. Bu, highway-env'de sorun:

```
Episode 1: Kolay trafik → ajan öğrenir "hızlan"
Episode 2: Zor trafik → "hızlan" → çarpışma → büyük ceza
Episode 3: Ajan öğrenir "yavaşla" → çok yavaş → düşük ödül
Episode 4: Kolay trafik ama ajan artık korkak → düşük ödül
→ Politika sallanır, kararsız
```

**2. High variance ortam:**
Highway-env'de her episode farklı trafik üretiyor. Bazı episode'lar kolay (boş yol),
bazıları zor (tıkanık trafik). PPO, son batch'teki deneyimlere göre güncelleme yapıyor.
Eğer son batch tesadüfen kolay episode'lardan oluşuyorsa, ajan "aşırı cesur" olur.
Sonra zor episode geldiğinde çarpar.

**3. Clip range yetersiz:**
Clip range=0.2 bile bu kadar değişken bir ortamda büyük politika değişimlerini
engelleyemiyor.

### DQN Neden Daha İyi?

**1. Off-policy + Experience Replay:**
DQN eski deneyimleri SAKLAR (50K deneyimlik buffer). Öğrenirken hem kolay hem zor
episode'lardan karışık örnekler çeker. Bu, politikanın tek bir senaryo tipine
overfitting yapmasını önler.

```
Buffer: [kolay_ep_1, zor_ep_1, kolay_ep_2, zor_ep_2, ...]
Her batch: rastgele 64 deneyim → dengeli öğrenme
```

**2. Target Network:**
Q-value hedefleri ayrı bir "donmuş" ağdan hesaplanır. Bu, öğrenme sırasında hedefin
sürekli kaymasını önler → daha kararlı.

**3. Epsilon Decay:**
Exploration yavaşça azalır — ajan yeni senaryolar keşfeder ama zamanla daha güvenilir
davranır. PPO'da exploration, entropy coefficient ile dolaylı kontrol edilir — daha az
öngörülebilir.

### Karşılaştırma Tablosu

```
┌─────────────────────┬──────────────────┬──────────────────┐
│       ÖZELLİK       │       PPO        │       DQN        │
├─────────────────────┼──────────────────┼──────────────────┤
│ Tip                 │ On-policy        │ Off-policy       │
│ Veri kullanımı      │ Bir kez kullan,  │ Buffer'da sakla, │
│                     │ sonra at         │ tekrar kullan    │
│ Action space        │ Continuous +     │ Sadece discrete  │
│                     │ Discrete         │                  │
│ Sample efficiency   │ Düşük            │ Yüksek           │
│ Stabilite (bu proje)│ Düşük (collapse) │ Yüksek           │
│ Hiperparametre      │ Daha az          │ Daha fazla       │
│ Karmaşıklık         │ Orta             │ Orta             │
│ Memory kullanımı    │ Düşük            │ Yüksek (buffer)  │
│ Highway-env sonucu  │ 62 ± 56 😢      │ 70 ± 40 😊      │
└─────────────────────┴──────────────────┴──────────────────┘
```

---

# 9. BUGS, ERRORS, AND HOW THEY WERE FIXED <a name="9-bugs"></a>

## Bug 1: env.configure() AttributeError

```
AttributeError: 'OrderEnforcing' object has no attribute 'configure'
```

**Ne oldu:** Highway-env'in eski versiyonlarında `env.configure(config)` ile ayar
yapılıyordu. Yeni versiyonlarda Gymnasium wrapper'ları (OrderEnforcing, PassiveEnvChecker)
ortamı sarıyor ve `.configure()` metodu erişilemez oluyor.

**Nasıl düzeltildi:** Config'i `gym.make()` zamanında geçtik:
```python
# YANLIŞ (eski API)
env = gym.make("highway-v0")
env.configure({"lanes_count": 4})

# DOĞRU (yeni API)
env = gym.make("highway-v0", config={"lanes_count": 4})
```

**Ders:** Kütüphane versiyonları önemli. API değişir, örnekler eski kalır.

## Bug 2: matplotlib tostring_rgb() Hatası

```
ValueError: cannot reshape array of size 0 into shape (...)
```

**Ne oldu:** Matplotlib'in yeni versiyonlarında `canvas.tostring_rgb()` çalışmıyor
veya boş array dönüyor.

**Nasıl düzeltildi:**
```python
# YANLIŞ (eski API)
img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# DOĞRU (yeni API)
img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
```

## Bug 3: PPO Policy Collapse

Yukarıda detaylı anlattım. Fully trained model half-trained'den kötü.

**Nasıl düzeltildi:** Algoritmayı PPO'dan DQN'e değiştirdik.

## Bug 4: Sabit Seed Video Problemi

**Ne oldu:** Tüm ajanlar seed=42 ile test ediliyordu. Bu seed zor bir trafik senaryosu
üretiyordu. Sonuç: üç aşamada da 18 frame — hepsi çarpıyordu. Video anlamsızdı.

**Nasıl düzeltildi:** `find_best_seed()` fonksiyonu eklendi. 13 farklı seed deneniyor,
her ajan için en uzun hayatta kalınan seed seçiliyor.

## Bug 5: Timeout (Eğitim çok uzun)

**Ne oldu:** İlk versiyonda 50K step, 256-unit ağ, 20 araç, 60 saniye duration
ile eğitim sunucu ortamında 10+ dakikada bitmiyordu.

**Nasıl düzeltildi:** Duration 40'a, araç sayısı 15'e düşürüldü. Yerel bilgisayarda
sorun yok çünkü timeout limiti yok.

---

# 10. WHAT I WOULD IMPROVE IN V2 <a name="10-v2"></a>

1. **Double DQN / Dueling DQN:** Standart DQN, Q-value'ları overestimate edebilir.
   Double DQN bunu düzeltir. SB3'te `DQN(..., policy_kwargs={"dueling": True})` ile
   kolayca eklenebilir.

2. **Prioritized Experience Replay:** Replay buffer'dan rastgele değil, "şaşırtıcı"
   deneyimleri öncelikli örneklemek. Nadir çarpışma deneyimleri daha sık öğrenilir.

3. **Learning Rate Scheduling:** Sabit 5e-4 yerine zamanla azalan learning rate.
   Başta hızlı öğren, sonra ince ayar yap.

4. **Multi-seed evaluation:** Final değerlendirmede 10 farklı seed'in ortalaması
   daha güvenilir bir metrik verir.

5. **Custom reward function:** Highway-env'in built-in ödülü yerine wrapper ile
   özel ödül fonksiyonu yazmak (mesela şerit değiştirme cezası, yakınlık cezası).

6. **Curriculum learning:** Önce az araçla (5 araç) eğit, sonra yavaşça artır (20).
   Bu, ajanın temel becerileri önce öğrenmesini sağlar.

7. **Hyperparameter sweep:** Optuna ile otomatik hiperparametre araması.

8. **TensorBoard logging:** Eğitim sırasında gerçek zamanlı izleme.
