
import os
import sqlite3
import pandas as pd
import numpy as np
import glob

biochem= pd.read_csv("./inputs/merged_biochem.csv")


biochem['compound_name'].unique()


compounds = ["Celecoxib", "Ketoprofen", "Mefenamic acid", "Lumiracoxib", "DMSO", 'Valdecoxib' ]


filtered_df = biochem[biochem["compound_name"].isin(compounds)][["plate", "well", "compound_name", "compound_concentration_um"]]
print(filtered_df["compound_name"].value_counts())


result_tables = {compound: filtered_df[filtered_df["compound_name"] == compound] for compound in compounds}


for compound, df in result_tables.items():
    print(f"Table pour {compound}:")
    print(df)
    print("\n" + "-"*50 + "\n")







df_celecoxib = result_tables["Celecoxib"]
df_ketoprofen = result_tables["Ketoprofen"]
df_mefenamic_acid = result_tables["Mefenamic acid"]
df_lumiracoxib = result_tables["Lumiracoxib"]
df_dmso = result_tables["DMSO"]



df_lumiracoxib['compound_concentration_um'].unique()
df_dmso



plate_id = "plate_41002898" 
df_dmso_filtered = df_dmso[df_dmso["plate"] == plate_id]

print(df_dmso_filtered["well"].tolist())






### put the name of you compound and it will extract all the data from sqlite files

def format_well(well):
    letter = well[0]
    number = well[1:]
    number = f"{int(number):02d}"
    return f"{letter}{number}"

def process_plate(plate_name, compound, result_tables, output_dir):
    try:
        db_path = f"/Users/grigoryanmariam/Library/Mobile Documents/com~apple~CloudDocs/Documents/thèse/Plates_rep1_2/{plate_name}.sqlite"
        if not os.path.exists(db_path):
            return pd.DataFrame()

        df_cpd = result_tables[compound]
        df_cpd_plate = df_cpd[df_cpd['plate'] == plate_name]
        df_cpd_plate['well'] = df_cpd_plate['well'].apply(format_well)
        wells = df_cpd_plate['well'].unique()

        conn = sqlite3.connect(db_path)
        tables = ["Cells", "Cytoplasm", "Nuclei"]

        well_filters = " OR ".join([f"LOWER(FileName_CellOutlines) LIKE '{well}_%'" for well in wells])
        query_image_filtered = f"""
        SELECT TableNumber, ImageNumber, FileName_CellOutlines
        FROM Image
        WHERE {well_filters}
        """

        df_images = pd.read_sql_query(query_image_filtered, conn)

        for table in tables:
            query = f"""
            SELECT {table}.*, Image_File.FileName_CellOutlines
            FROM {table}
            JOIN ({query_image_filtered}) AS Image_File 
            USING (TableNumber, ImageNumber);
            """
            df = pd.read_sql_query(query, conn)
            output_file = os.path.join(output_dir, f"{table}_filtered_with_filenames_{plate_name}.parquet")
            df.to_parquet(output_file, index=False)

        conn.close()

        df_combined = pd.concat([pd.read_parquet(os.path.join(output_dir, f"{table}_filtered_with_filenames_{plate_name}.parquet")) for table in tables], axis=1)
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

        df_combined['well'] = df_combined['FileName_CellOutlines'].str.extract(r'(^[A-Z0-9]+)')[0]
        df_combined['well'] = df_combined['well'].apply(format_well)

        df_combined = df_combined.merge(
            df_cpd_plate[['well', 'compound_name', 'compound_concentration_um', 'plate']],
            on='well',
            how='left'
        )

        df_combined['plate'] = df_combined['plate'].fillna('unknown')
        df_combined = df_combined.drop(columns=['TableNumber', 'ImageNumber', 'FileName_CellOutlines'])
        
        return df_combined

    except Exception as e:
        return pd.DataFrame()



compound = "Valdecoxib"
output_dir = "/Users/grigoryanmariam/Downloads"
plate_names = result_tables[compound]['plate']

df_final = pd.concat([process_plate(plate_name, compound, result_tables, output_dir) for plate_name in plate_names])


df_final['plate'] = df_final['plate'].str.extract(r'(\d{4})$')

print(df_final['plate'].nunique())

rep1_plates = ['2695', '2702', '2689', '2701', '2693', '2697', '2699', '2690']

df_final['replicate'] = np.where(df_final['plate'].isin(rep1_plates), '1', '2')

print(df_final.head())



df_final.to_csv("./outputs/Valdecoxib.csv", index=False)

