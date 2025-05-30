import pickle
import numpy as np

def view_embedding_values(pkl_file_path="face_recognition_database.pkl", max_samples=3):
    """
    Fungsi untuk melihat dan menampilkan nilai-nilai numerik dari embedding 128 dimensi
    tanpa visualisasi grafik
    
    Parameters::
    - pkl_file_path: Path ke file pickle yang berisi data embedding
    - max_samples: Jumlah maksimum sampel yang akan ditampilkan per nama (default: 3)
    """
    print(f"Membuka file: {pkl_file_path}")
    
    try:
        # Membuka file pickle
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Memastikan data adalah dictionary dan memiliki embedding
        if not isinstance(data, dict) or 'embeddings' not in data or 'names' not in data:
            print("Format data tidak sesuai. Diperlukan dictionary dengan kunci 'embeddings' dan 'names'")
            return
            
        # Ambil data embedding dan nama
        embeddings = data['embeddings']
        names = data['names']
        
        # Periksa dimensi embedding
        if len(embeddings) == 0:
            print("Tidak ada data embedding")
            return
            
        first_embedding = embeddings[0]
        embedding_dim = first_embedding.shape[0] if hasattr(first_embedding, 'shape') else len(first_embedding)
        
        print(f"\n===== INFO EMBEDDING =====")
        print(f"Jumlah total embedding: {len(embeddings)}")
        print(f"Dimensi embedding: {embedding_dim}")
        
        # Jika dimensi bukan 128, berikan peringatan
        if embedding_dim != 128:
            print(f"PERHATIAN: Embedding memiliki dimensi {embedding_dim}, bukan 128!")
        
        # Konversi ke numpy array untuk memudahkan analisis
        embeddings_array = np.array(embeddings)
        
        # Statistik data embedding
        print(f"\n===== STATISTIK EMBEDDING =====")
        print(f"Min: {embeddings_array.min():.4f}")
        print(f"Max: {embeddings_array.max():.4f}")
        print(f"Mean: {embeddings_array.mean():.4f}")
        print(f"Std: {embeddings_array.std():.4f}")
        
        # Daftar nama unik
        unique_names = list(set(names))
        print(f"\n===== DAFTAR NAMA =====")
        print(f"Jumlah nama unik: {len(unique_names)}")
        for i, name in enumerate(unique_names):
            count = names.count(name)
            print(f"{i+1}. {name}: {count} sampel")
        
        # Tampilkan nilai embedding untuk setiap nama
        print(f"\n===== NILAI EMBEDDING PER NAME =====")
        for name in unique_names:
            print(f"\nNama: {name}")
            
            # Dapatkan indeks untuk nama ini
            indices = [i for i, n in enumerate(names) if n == name]
            samples_to_show = min(max_samples, len(indices))
            
            for sample_idx, idx in enumerate(indices[:samples_to_show]):
                print(f"  Sampel {sample_idx+1} (Index {idx}):")
                embedding = embeddings[idx]
                
                # Tampilkan nilai embedding dalam format yang lebih mudah dibaca
                print("    [", end="")
                for i, value in enumerate(embedding):
                    print(f"{value:.6f}", end="")
                    # Tambahkan koma kecuali untuk elemen terakhir
                    if i < len(embedding) - 1:
                        print(", ", end="")
                        # Buat baris baru setiap 8 nilai untuk keterbacaan
                        if (i + 1) % 8 == 0:
                            print("\n     ", end="")
                print("]")
                
            # Jika ada lebih banyak sampel yang tidak ditampilkan
            if len(indices) > samples_to_show:
                print(f"  ... dan {len(indices) - samples_to_show} sampel lainnya tidak ditampilkan")
        
        # Hitung dan tampilkan embedding rata-rata untuk setiap nama
        print(f"\n===== EMBEDDING RATA-RATA PER NAMA =====")
        for name in unique_names:
            indices = [i for i, n in enumerate(names) if n == name]
            mean_embedding = np.mean([embeddings[i] for i in indices], axis=0)
            
            print(f"\nNama: {name} (rata-rata dari {len(indices)} sampel)")
            # Tampilkan nilai embedding rata-rata
            print("  [", end="")
            for i, value in enumerate(mean_embedding):
                print(f"{value:.6f}", end="")
                # Tambahkan koma kecuali untuk elemen terakhir
                if i < len(mean_embedding) - 1:
                    print(", ", end="")
                    # Buat baris baru setiap 8 nilai untuk keterbacaan
                    if (i + 1) % 8 == 0:
                        print("\n   ", end="")
            print("]")
            
    except Exception as e:
        print(f"Error: {e}")

# Contoh penggunaan:
if __name__ == "__main__":
    view_embedding_values("face_recognition_database.pkl", max_samples=2)