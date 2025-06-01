import sqlite3

# Veritabanı dosyasını oluştur ve bağlan
conn = sqlite3.connect("urun_bilgileri.db")
cursor = conn.cursor()

# Eğer tablo daha önce oluşturulmadıysa oluştur
cursor.execute("""
CREATE TABLE IF NOT EXISTS urun_bilgileri (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    baslik TEXT,
    icerik TEXT
)
""")

# Örnek veri ekleyelim
veriler = [
    ("Kargo Süreci", "Kargom 3 gün içinde elime ulaştı, ancak gecikme yaşanabilir."),
    ("İade Politikası", "Ürünü 14 gün içinde koşulsuz iade edebilirsiniz."),
    ("Ödeme Seçenekleri", "Kredi kartı, havale ve kapıda ödeme seçenekleri mevcut."),
    ("Teknik Destek", "Ürün çalışmıyorsa destek hattını arayabilirsiniz."),
    ("İletişim", "Müşteri hizmetlerine haftaiçi 09:00 - 18:00 arası ulaşabilirsiniz.")
]

cursor.executemany("INSERT INTO urun_bilgileri (baslik, icerik) VALUES (?, ?)", veriler)

# Değişiklikleri kaydet ve bağlantıyı kapat
conn.commit()
conn.close()

print("Veritabanı ve tablo başarıyla oluşturuldu ve veriler eklendi.")
