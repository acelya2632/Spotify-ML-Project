using CSV
using DataFrames
using StatsBase

raw_path = raw"C:\CORUNA\MachineLearning\Project\Project\Spotify-ML-Project\datasets\Data\Raw\tracks.csv"
df = CSV.read(raw_path, DataFrame; select=["popularity"])

############################################################
# PAS 2 – CLEAN: missing, duplicate, range, sanity check
############################################################

println("\n=== PAS 2: Curățare date ===")

## 2.1. Eliminare valori lipsă (missing) în popularity
n_missing = count(ismissing, df.popularity)
println("Valori lipsă în popularity: $n_missing")

if n_missing > 0
    df = dropmissing(df, :popularity)
    println("După dropmissing: rânduri = $(nrow(df))")
end

## 2.2. Eliminare duplicate (OPȚIONAL)
# Atenție: cu o singură coloană, asta păstrează doar valorile distincte.
# Pentru ETL de modelare, de obicei PĂSTREZI duplicatele, deci las opțional.

remove_duplicates = false  # pune true dacă chiar vrei doar valori unice

if remove_duplicates
    before = nrow(df)
    unique!(df)  # elimină rânduri identice
    println("Duplicate eliminate: $(before - nrow(df)) rânduri")
end

## 2.3. Validare range: 0 ≤ popularity ≤ 100

min_pop = minimum(df.popularity)
max_pop = maximum(df.popularity)
println("Range inițial popularity: [$min_pop, $max_pop]")

df = filter(row -> 0 ≤ row.popularity ≤ 100, df)

min_pop = minimum(df.popularity)
max_pop = maximum(df.popularity)
println("Range după filtrare: [$min_pop, $max_pop]")
println("Rânduri după curățare: $(nrow(df))")

## 2.4. (Opțional) Sanity check pe distribuție

println("\nSanity check – statistici de bază:")
display(describe(select(df, :popularity)))

# Histogramă numerică foarte simplă – număr de exemple pe intervale de 10
bins = floor.(df.popularity ./ 10)  # 0-9 => 0, 10-19 => 1, etc.
pop_counts = countmap(bins)

println("\nNumăr de piese pe intervale de 10 popularitate (0 => 0–9, 1 => 10–19, etc.):")
for (bin, cnt) in sort(collect(pop_counts); by=first)
    println("Interval $(bin*10)-$(bin*10+9): $cnt piese")
end

############################################################
# PAS 3 – TRANSFORM: creare target_binary (hit / non-hit)
############################################################

println("\n=== PAS 3: Creare target_binary (popularity > 50) ===")

# 1 = hit, 0 = non-hit
df.target_binary = ifelse.(df.popularity .> 50, 1, 0)

n_hit  = sum(df.target_binary .== 1)
n_non  = sum(df.target_binary .== 0)
total  = nrow(df)

println("Total piese: $total")
println("Non-hit (0): $n_non")
println("Hit (1)    : $n_hit")

############################################################
# PAS 4 – SELECT: dataset specific pentru Approach 1
############################################################

println("\n=== PAS 4: Selectare coloane pentru Approach 1 ===")

# Pentru moment: un singur feature + target
df_approach1 = select(df, :popularity, :target_binary)

println("Primele 5 rânduri din datasetul pentru approach1:")
display(first(df_approach1, 5))

############################################################
# PAS 5 – NORMALIZARE (OPȚIONALĂ, recomandată)
############################################################

println("\n=== PAS 5: Normalizare Min-Max pe popularity (opțional) ===")

# Calculăm parametrii Min-Max
pop_min = minimum(df_approach1.popularity)
pop_max = maximum(df_approach1.popularity)

println("Parametri Min-Max popularity: min=$pop_min, max=$pop_max")

# Creăm o coloană nouă normalizată, nu suprascriem popularitatea brută
df_approach1.popularity_minmax = (df_approach1.popularity .- pop_min) ./ (pop_max - pop_min)

# (Dacă vrei standardizare Zero-Mean în loc):
# μ = mean(df_approach1.popularity)
# σ = std(df_approach1.popularity)
# df_approach1.popularity_zeromean = (df_approach1.popularity .- μ) ./ σ

println("Primele 5 rânduri după normalizare:")
display(first(df_approach1, 5))

############################################################
# PAS 6 – QUALITY CHECK (rapid după transformări)
############################################################

println("\n=== PAS 6: Quality check final ===")

println("Describe pe popularity brută:")
display(describe(select(df_approach1, :popularity)))

println("\nDescribe pe popularity_minmax:")
display(describe(select(df_approach1, :popularity_minmax)))

println("\nDistribuție target_binary:")
println("Non-hit (0): $(sum(df_approach1.target_binary .== 0))")
println("Hit    (1):  $(sum(df_approach1.target_binary .== 1))")

############################################################
# PAS 7 – LOAD: salvare fișiere în Clean/approach1
############################################################

println("\n=== PAS 7: Salvare fișiere CSV pentru Approach 1 ===")

# 1) Dataset „raw” pentru model (popularity + target)
path_raw = joinpath(approach1_dir, "approach1_raw.csv")
CSV.write(path_raw, select(df_approach1, :popularity, :target_binary))
println("Salvat: $path_raw")

# 2) Dataset cu coloana normalizată (poate fi folosit direct pentru ML)
path_norm = joinpath(approach1_dir, "approach1_minmax.csv")
CSV.write(path_norm, select(df_approach1, :popularity_minmax, :target_binary))
println("Salvat: $path_norm")

# 3) (Opțional) salvăm parametrii de normalizare într-un fișier simplu TXT
params_path = joinpath(approach1_dir, "normalization_minmax_params.txt")
open(params_path, "w") do io
    write(io, "Min-Max parameters for popularity\n")
    write(io, "min = $pop_min\n")
    write(io, "max = $pop_max\n")
end
println("Salvat parametri normalizare: $params_path")

println("\n=== ETL Approach 1 finalizat. ===")

