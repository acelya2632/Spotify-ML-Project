using CSV
using DataFrames
using StatsBase

raw_path = raw"C:\CORUNA\MachineLearning\Project\Project\Spotify-ML-Project\datasets\Data\Raw\tracks.csv"
df = CSV.read(raw_path, DataFrame; select=["popularity"])

println("\n=== PAS 2: Curățare date ===")

# 2.1 Missing
n_missing = count(ismissing, df.popularity)
println("Valori lipsă: $n_missing")
if n_missing > 0
    df = dropmissing(df, :popularity)
end

# 2.3 Range check
df = filter(row -> 0 ≤ row.popularity ≤ 100, df)

println("Range după filtrare: ",
        minimum(df.popularity), " - ", maximum(df.popularity))

# 2.4 Describe
display(describe(select(df, :popularity)))

# Histogram
bins = floor.(df.popularity ./ 10)
pop_counts = countmap(bins)
println("\nDistribuție pe intervale:")
for (bin, cnt) in sort(collect(pop_counts); by=first)
    println("Interval $(bin*10)-$(bin*10+9): $cnt")
end

############################################################
# PAS 3 – Creare TARGET MULTICLASS (0–3)
############################################################

println("\n=== PAS 3: Creare target_multi (4 clase) ===")

function categorize_popularity(p)
    if p ≤ 30
        return 0
    elseif p ≤ 60
        return 1
    elseif p ≤ 80
        return 2
    else
        return 3
    end
end

df.target_multi = categorize_popularity.(df.popularity)

println("Număr pe clase:")
println("0 (Niche):    ", sum(df.target_multi .== 0))
println("1 (Average):  ", sum(df.target_multi .== 1))
println("2 (Popular):  ", sum(df.target_multi .== 2))
println("3 (Viral):    ", sum(df.target_multi .== 3))

############################################################
# PAS 4 – Dataset final
############################################################

df_approach2 = select(df, :popularity, :target_multi)

println("\nPrimele 5 rânduri:")
display(first(df_approach2, 5))

############################################################
# PAS 5 – NORMALIZARE (OPȚIONALĂ, recomandată)
############################################################
println("\n=== PAS 5: Normalizare Min-Max pe popularity (opțional) ===")

pop_min = minimum(df_approach2.popularity)
pop_max = maximum(df_approach2.popularity)

println("Parametri Min-Max popularity: min=$pop_min, max=$pop_max")

df_approach2[:, :popularity_minmax] =
    (df_approach2.popularity .- pop_min) ./ (pop_max - pop_min)

println("Primele 5 rânduri după normalizare:")
display(first(df_approach2, 5))



############################################################
# PAS 6 – Salvare fișier final
############################################################

save_path = raw"C:\CORUNA\MachineLearning\Project\Project\Spotify-ML-Project\datasets\Data\Clean\approach2.csv"
CSV.write(save_path, df_approach2)

println("Salvat la: $save_path")
println("\n=== ETL Approach 2 finalizat ===")
