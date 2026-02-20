# FastCNN (MNIST)

Bu repo, size verdiğiniz mimariyi PyTorch ile uygulayan basit bir örnek sağlar.

Varsayımlar
- MaxPool: kernel_size=2, stride=2 (girdi: 28x28x1)
- Model sonu `Softmax` döndürür. Eğitimde `CrossEntropyLoss` kullanmak için softmax'i modelden çıkarmak daha uygundur.

Dosyalar
- `model.py`: `FastCNN` sınıfı.
- `train.py`: MNIST verisi ile örnek eğitim döngüsü.
- `requirements.txt`: Gerekli paketler.

Hızlı başlatma
1. Sanal ortam oluşturun ve aktivasyon yapın.
2. Paketleri yükleyin:

```bash
pip install -r requirements.txt
```

3. Eğitim çalıştırın (örnek 2 epoch):

```bash
python train.py
```
