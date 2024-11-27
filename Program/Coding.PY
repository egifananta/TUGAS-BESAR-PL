print("MK Pemrograman Lanjut")
print("Eggy Alfan Ananta-1103223194")
print("Aisha Patricia-1103223067")
print("Velma Beryl Anzalia-1103223038")
print("Faza Salsabila Rahayu Wilujeng-1103223008")

import pandas as pd

# Data mahasiswa
data = {
    "Nama": [
        "Eggy Alfan Ananta",
        "Aisha Patricia",
        "Velma Beryl Anzalia",
        "Faza Salsabila Rahayu Wilujeng"
    ],
    "NIM": [
        "1103223194",
        "1103223067",
        "1103223038",
        "1103223008"
    ]
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Menampilkan informasi mata kuliah
print("MK Pemrograman Lanjut")
print(df)
