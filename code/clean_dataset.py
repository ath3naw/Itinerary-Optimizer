import pandas as pd
import utm
def clean_dataset(start_lat, start_lon, end_lat, end_lon):
    df = pd.read_csv('data/NYC_museums.csv')

    # Only need name and location
    cols = [
        "the_geom",
        "NAME"
    ]

    df = df[cols].dropna()

    df = df.rename(columns={
        "NAME": "name",
        "ADDRESS2": "address",
        "COST": "cost",
    })

    coords = (
        df["the_geom"]
        .str.replace("POINT (", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.split(" ", expand=True)
    )

    df["lon"] = coords[0].astype(float)
    df["lat"] = coords[1].astype(float)

    df["indoor"] = True
    df["museum"] = True

    coords = [
        utm.from_latlon(lat, lon)
        for lat, lon in zip(df["lat"], df["lon"])
    ]

    df[["easting", "northing", "zone_number", "zone_letter"]] = coords
    df = df.drop(columns=["the_geom", "zone_number", "zone_letter"])

    data = {"name": ["start", "end"], "lat": [float(start_lat), float(end_lat)], "lon": [float(start_lon), float(end_lon)],
            "easting": [utm.from_latlon(float(start_lat), float(start_lon))[0], utm.from_latlon(float(end_lat), float(end_lon))[0]],
            "northing": [utm.from_latlon(float(start_lat), float(start_lon))[1], utm.from_latlon(float(end_lat), float(end_lon))[1]],}
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    return df