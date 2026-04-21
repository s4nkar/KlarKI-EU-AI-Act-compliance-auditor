import os

data_dir = "data/regulatory"

keep_ai_act = {
    "article_3.txt", "article_5.txt", "article_6.txt", 
    "article_8.txt", "article_9.txt", "article_10.txt", 
    "article_11.txt", "article_12.txt", "article_13.txt", 
    "article_14.txt", "article_15.txt", "article_50.txt",
    "annex_iii.txt", "annex_iv.txt"
}

keep_gdpr = {
    "article_5.txt", "article_6.txt", "article_9.txt",
    "article_13.txt", "article_14.txt", "article_15.txt",
    "article_16.txt", "article_17.txt", "article_18.txt",
    "article_19.txt", "article_20.txt", "article_21.txt",
    "article_22.txt", "article_24.txt", "article_25.txt",
    "article_30.txt", "article_32.txt", "article_35.txt",
    "article_44.txt", "article_45.txt", "article_46.txt",
    "article_47.txt", "article_48.txt", "article_49.txt"
}

deleted = 0

ai_dir = os.path.join(data_dir, "eu_ai_act")
if os.path.exists(ai_dir):
    for f in os.listdir(ai_dir):
        if f.endswith(".txt") and f not in keep_ai_act:
            os.remove(os.path.join(ai_dir, f))
            print(f"Deleted {f}")
            deleted += 1

gdpr_dir = os.path.join(data_dir, "gdpr")
if os.path.exists(gdpr_dir):
    for f in os.listdir(gdpr_dir):
        if f.endswith(".txt") and f not in keep_gdpr:
            os.remove(os.path.join(gdpr_dir, f))
            print(f"Deleted {f}")
            deleted += 1

print(f"Deleted {deleted} unnecessary files.")
