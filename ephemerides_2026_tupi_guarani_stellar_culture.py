#!/usr/bin/env python3
"""
ephemerides_2026_tupi_guarani_stellar_culture.py

Cultural astronomy visualization tool that generates a multi-night sky-dome animation 
based on the 2026 ephemerides table referenced to the Tupi-Guarani astronomical culture, 
compiled by the Seichú astronomical observatory, located in the Parque Arqueológico da Pedra do Sol, 
in the Serra do Cipó, Minas Gerais state.


Copyright (C) 2025  Seichú Astronomical Observatory
Parque Arqueológico da Pedra do Sol, Serra do Cipó, Minas Gerais, Brazil
Website: <https://observatorioseichu.com.br/>.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from tqdm import tqdm
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm

from PIL import Image 
import textwrap 

# Custom font and style  
plt.rcParams.update({
    "font.family": "serif",            
    "font.serif": ["DejaVu Serif"],    
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# ---------- CONFIG ----------
INPUT_CSV = "stars.csv"
OUTPUT_GIF = "ephemerides_2026_tupi_guarani_stellar_culture.gif"
OBSERVER_LAT = -15.8
OBSERVER_LON = -47.8
OBSERVER_HEIGHT_M = 1100.0
OBSERVER_TZ = "America/Sao_Paulo"

START_HOUR_LOCAL = 18
NIGHT_DURATION_H = 10.0
FRAMES_PER_NIGHT = 24
FPS = 10
MAX_STARS_TO_LABEL = 200
SHOW_ONLY_THIS_DATE_LIST = None

# Map constellation names to images  
CONST_IMAGES = {
    'Estrela Vagalume (Uam-rana)': 'images/vagalume.png',
    'Estrela Tinguaçu (Tinguaçu)': 'images/tinguacu.png',
    'Pata da Ema (Nhandu pysá)': 'images/pata-da-ema.png',
    'Ovos de Pássaro (Guirá rupiá)': 'images/dois-ovos.png',
    'Colibri (Mainamy)': 'images/colibri.png',
    'Canoa (Yar Ragapaw)': 'images/canoa.png',
    'Jabuti (Zauxihu Ragapaw)': 'images/jabuti.png',
    'Ema (Guirá Nhandú)':'images/ema.png',
    'Onça Celeste (Xivi / Charia)':'images/onca.png',
    'Cobra de Fogo (Mboi Tatá)':'images/cobra.png',
    'Seriema (Azim)':'images/seriema.png',
    'Cervo (Guaśú / Suçu-Guasu)':'images/cervo.png',
    'Anta do Norte (Tapí’i)':'images/tapir.png',
    'Poço do Caititu (Coxi Huguá)':'images/caititu.png',
    'Arapuca (Aka’ekorá / Seichu Jurá)':'images/arapuca.png',
    'Plêiades (Seichú)':'images/pleiades.png',
    'Queixada da Anta (Tapí’i Rainhykä)':'images/queixada.png',
    'Homem Velho (Tuiváe)':'images/homen_velho.png',
    'Poço do Tapir (Tapí’i Huguá)':'images/tapir-poco.png',
    'Encantado da Fertilidade (Joykexo)':'images/joykexo.png',
}

# ---------- HELPERS ----------
# --- Converts RA written as hours:minutes:seconds to decimal degrees ---
def parse_hms_to_deg(hms_str):
    s = hms_str.replace('h', ':').replace('m', ':').replace('s','').replace(' ','')
    
    parts = [float(p) for p in s.replace(':',' ').split()]
    
    hh, mm, ss = (parts + [0,0])[:3]
   
    return (hh + mm/60 + ss/3600)*15.0

# --- Converts Dec written as hours:minutes:seconds to decimal degrees ---
def parse_dms_to_deg(dms_str):
    if not isinstance(dms_str, str) or dms_str.strip() == '':
        return np.nan
    
    s = dms_str.replace('−','-').replace('°',':').replace('′',':').replace('\'',':')\
                .replace('″','').replace('"','').replace(' ','')
    
    parts = [p for p in s.replace(':',' ').split() if p != '']
    
    sign = -1 if parts[0].startswith('-') else 1
    if parts[0].startswith('-'):
        parts[0] = parts[0][1:]
    
    d, m, s_ = (list(map(float, parts)) + [0,0])[:3]
    
    return sign * (d + m/60 + s_/3600)

# --- Reads the star data from CSV and creates a clean DataFrame---
def df_from_csv(path):
    df = pd.read_csv(path, dtype=str)
    
    colmap = {c.lower().strip(): c for c in df.columns}

    df2 = pd.DataFrame()
    df2['date'] = df[colmap['date']].str.strip()
    df2['star'] = df[colmap['star']].str.strip()
    df2['constellation'] = df[colmap['constellation']].str.strip()
    df2['RA_hms'] = df[colmap['ra_hms']].str.strip()
    df2['Dec_dms'] = df[colmap['dec_dms']].str.strip()
   
    df2['mag'] = df[colmap['mag']].astype(float) if 'mag' in colmap else np.nan

    df2['RA_deg'] = df2['RA_hms'].apply(parse_hms_to_deg)
    df2['Dec_deg'] = df2['Dec_dms'].apply(parse_dms_to_deg)
    
    return df2

# ---------- MAIN ----------
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} not found.")
    df = df_from_csv(INPUT_CSV)

    if SHOW_ONLY_THIS_DATE_LIST:
        df = df[df['date'].isin(SHOW_ONLY_THIS_DATE_LIST)]

    unique_dates = list(dict.fromkeys(df['date'].tolist()))
    n_nights = len(unique_dates)
    print(f"Found {len(df)} stars on {n_nights} nights.")

    location = EarthLocation(lat=OBSERVER_LAT*u.deg, lon=OBSERVER_LON*u.deg, height=OBSERVER_HEIGHT_M*u.m)
    tz = ZoneInfo(OBSERVER_TZ)

    idx_by_date = {date: df.index[df['date']==date].tolist() for date in unique_dates}

    # Frame times
    all_frame_times_utc = []
    frame_to_date = []
    for date_str in unique_dates:
        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        start_local = datetime(date_obj.year, date_obj.month, date_obj.day, START_HOUR_LOCAL, 0, 0, tzinfo=tz)
        frame_offsets = np.linspace(0, NIGHT_DURATION_H, FRAMES_PER_NIGHT)
        for off_hr in frame_offsets:
            dt_local = start_local + timedelta(hours=float(off_hr))
            dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
            all_frame_times_utc.append(Time(dt_utc))
            frame_to_date.append(date_str)

    total_frames = len(all_frame_times_utc)
    print(f"Total frames: {total_frames}")

    # Sky coordinates
    skycoords = SkyCoord(ra=df['RA_deg'].values*u.deg, dec=df['Dec_deg'].values*u.deg, frame='icrs')

    alt_arr = np.zeros((total_frames, len(df))) * u.deg
    az_arr = np.zeros((total_frames, len(df))) * u.deg

    print("Computing alt/az for every frame...")
    for i, t in enumerate(tqdm(all_frame_times_utc, unit="frame")):
        altaz = skycoords.transform_to(AltAz(obstime=t, location=location))
        alt_arr[i,:] = altaz.alt
        az_arr[i,:] = altaz.az

    # --- Assign colors per constellation ---
    from matplotlib import cm

    constellations = df['constellation'].unique()
    n_const = len(constellations)
    cmap = cm.get_cmap('tab10')
    const_colors = {c: cmap(i / max(n_const-1, 1)) for i, c in enumerate(constellations)}

    # --- Animation ---
    fig = plt.figure(figsize=(10,9), facecolor="#d1cfb6")
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor("#b6d1bd")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0,90)
    ax.set_position([0.04, 0.05, 0.64, 0.90])  # left, bottom, width, height (figure fraction)
    
    legend_ax = fig.add_axes([0.7, 0.05, 0.28, 0.90])  # left, bottom, width, height (figure fraction)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')
    

    # preload and downscale images
    images_cache = {}
    max_img_size = (80, 80)  

    for name, path in CONST_IMAGES.items():
        if os.path.exists(path):
            try:
                img = Image.open(path)      
                img.thumbnail(max_img_size) 
                images_cache[name] = np.array(img)  
            except Exception:
                images_cache[name] = None


    def update(frame):
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0,90)
        
        # Current time text
        tod_local = all_frame_times_utc[frame].to_datetime(timezone=tz)
        ax.text(0.5, 1.05, tod_local.strftime("%Y-%m-%d %H:%M %Z\n"), transform=ax.transAxes, ha='center',fontsize=11)

        current_date = frame_to_date[frame]
        indices = idx_by_date[current_date]

        start_frame = frame - (frame % FRAMES_PER_NIGHT)  # first frame of the night


        current_date = frame_to_date[frame]
        indices = idx_by_date[current_date]
        start_frame = frame - (frame % FRAMES_PER_NIGHT)

        # --- Polar grid lines (same as before) ---
        for deg in range(0, 360, 90):
            ax.plot([np.deg2rad(deg), np.deg2rad(deg)], [0, 90], color="#775f5f", lw=1.5, linestyle='-', alpha=0.8)
        for deg in range(0, 360, 15):
            if deg % 90 == 0:
                continue
            ax.plot([np.deg2rad(deg), np.deg2rad(deg)], [0, 90], color="#a79595", lw=0.7, linestyle='--', alpha=0.6)

        # --- Draw stars and traces (same as before) ---
        for idx in indices:
            color = const_colors[df.loc[idx,'constellation']]

            r_traj = 90 - alt_arr[start_frame:frame+1, idx].to(u.deg).value
            az_traj = az_arr[start_frame:frame+1, idx].to(u.deg).value
            ax.plot(np.deg2rad(az_traj), np.clip(r_traj,0,90), color=color, alpha=0.5, linewidth=1)

            r_curr = 90 - alt_arr[frame, idx].to(u.deg).value
            az_curr = az_arr[frame, idx].to(u.deg).value
            size = np.clip((8 - df.loc[idx, 'mag'])**2, 8, 200) if 'mag' in df.columns and not np.isnan(df.loc[idx,'mag']) else 50
            ax.scatter(np.deg2rad(az_curr), np.clip(r_curr,0,90), s=size, color=color, alpha=1.0)

            if idx < MAX_STARS_TO_LABEL:
                ax.text(np.deg2rad(az_curr), np.clip(r_curr-5,0,90),
                        f"{df.loc[idx,'star']}",
                        fontsize=8, ha='center', va='top', alpha=0.7)

        # --- Legend panel ---
        legend_ax.clear()
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis('off')

        # Title
        legend_ax.text(
            0.5, 0.98,
            "Stellar Ephemerides 2026\nTupi-Guarani Stellar Culture",
            ha='center', va='top',
            fontsize=12, fontweight='bold'
        )
        
        desc_text = ("Visualisation by the Seichú Astronomical Observatory, relying on ‘Efemérides Estelares, 2026 - Cultura Estelar Tupi-Guarani’ compiled by the Seichú Astronomical Observatory, Serra do Cipó, Minas Gerais, Brazil")
        wrapped_text = "\n".join(textwrap.wrap(desc_text, width=40))
        legend_ax.text(
            0.5, 0.91,
            wrapped_text,
            ha='center', va='top',
            fontsize=9, wrap=True
        )

        """         
        # Current time
        tod_local = all_frame_times_utc[frame].to_datetime(timezone=tz)
        legend_ax.text(
            0.5, 0.78,
            tod_local.strftime("%Y-%m-%d %H:%M %Z"),
            ha='center', va='top',
            fontsize=11
        )
        """
        # Constellation images + labels
        consts_today = sorted(df.loc[indices, 'constellation'].unique()) if len(indices) > 0 else []
        n = len(consts_today)
        if n > 0:
            top = 0.66  # start below time text
            bottom = 0.05
            avail = top - bottom
            step = min(0.18, avail / max(n, 1))

        for i, const_name in enumerate(consts_today):
            y = top - i * step

            if const_name in images_cache and images_cache[const_name] is not None:
                img = images_cache[const_name]
                im = OffsetImage(img, zoom=1.0)
                ab = AnnotationBbox(im, (0.5, y), xycoords='axes fraction', frameon=False)
                legend_ax.add_artist(ab)

            legend_ax.text(
                0.5, y - step * 0.6,
                const_name,
                transform=legend_ax.transAxes,
                fontsize=10,
                ha='center',
                va='top',
                color=const_colors.get(const_name, 'black'),
                fontweight='bold'
            )

        return ax,

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/FPS, blit=False)

    print(f"Saving animation to {OUTPUT_GIF} ...")
    Writer = animation.PillowWriter(fps=FPS)
    ani.save(OUTPUT_GIF, writer=Writer, dpi=80)
    print("Saved. Done.")

if __name__ == "__main__":
    main()
