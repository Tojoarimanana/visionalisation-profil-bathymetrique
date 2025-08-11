import streamlit as st
import numpy as np
import pyvista as pv
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline
from geopy.distance import geodesic
import pyproj
import folium
from streamlit_folium import st_folium
import tempfile
import pandas as pd
import rasterio
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import base64
import os
from scipy.ndimage import zoom
import math

# Activer le mode "wide" pour maximiser l'utilisation de l'écran
st.set_page_config(layout="wide")

# --- Extraction des coordonnées depuis un objet pyvista ---
def extract_coords_from_mesh(mesh):
    try:
        points = mesh.points
        if len(points) == 0:
            raise ValueError("Le fichier VTK ne contient aucun point.")
        return points
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des points VTK : {e}")
        return None

# --- Conversion forcée des coordonnées UTM vers lat/lon (zone 38S) ---
def convert_utm_to_latlon(points):
    try:
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
        st.info("Conversion forcée des coordonnées UTM zone 38S en lat/lon...")
        transformer = pyproj.Transformer.from_crs(
            crs_from="epsg:32738",  # UTM zone 38S WGS84
            crs_to="epsg:4326",     # WGS84 lat/lon
            always_xy=True,
        )
        lon, lat = transformer.transform(X, Y)
        if np.any(Z > 0):
            st.warning(f"Attention : {np.sum(Z > 0)} points VTK ont des profondeurs positives (attendu : négatif pour les profondeur sous le niveau de la mer).")
        return lon, lat, Z
    except Exception as e:
        st.error(f"Erreur lors de la conversion UTM vers lat/lon : {e}")
        return None, None, None

# --- Calcul distances cumulées ---
def compute_cumulative_distances(lon, lat):
    try:
        coords = list(zip(lat, lon))
        distances = [0]
        for i in range(1, len(coords)):
            d = geodesic(coords[i-1], coords[i]).meters
            distances.append(distances[-1] + d)
        return np.array(distances)
    except Exception as e:
        st.error(f"Erreur lors du calcul des distances cumulées : {e}")
        return None

# --- Extension des coordonnées lon/lat et Z ---
def extend_coordinates(lon, lat, Z, distances, extension_ratio):
    try:
        total_length = distances[-1]
        extension_length = total_length * extension_ratio / 2

        delta_lon = lon[-1] - lon[-2] if len(lon) >= 2 else 0
        delta_lat = lat[-1] - lat[-2] if len(lat) >= 2 else 0
        segment_length = geodesic((lat[-1], lon[-1]), (lat[-2], lon[-2])).meters if len(lon) >= 2 else 1.0
        if segment_length > 0:
            delta_lon /= segment_length
            delta_lat /= segment_length
        else:
            delta_lon, delta_lat = 0, 0

        num_points_end = int(extension_length / (total_length / (len(lon) - 1))) + 1
        extended_lon_end = [lon[-1] + delta_lon * (i * extension_length / num_points_end) for i in range(1, num_points_end + 1)]
        extended_lat_end = [lat[-1] + delta_lat * (i * extension_length / num_points_end) for i in range(1, num_points_end + 1)]
        extended_Z_end = [Z[-1]] * num_points_end
        extended_dist_end = [distances[-1] + i * extension_length / num_points_end for i in range(1, num_points_end + 1)]

        delta_lon_start = lon[1] - lon[0] if len(lon) >= 2 else 0
        delta_lat_start = lat[1] - lat[0] if len(lat) >= 2 else 0
        segment_length_start = geodesic((lat[1], lon[1]), (lat[0], lon[0])).meters if len(lon) >= 2 else 1.0
        if segment_length_start > 0:
            delta_lon_start /= segment_length_start
            delta_lat_start /= segment_length_start
        else:
            delta_lon_start, delta_lat_start = 0, 0

        num_points_start = int(extension_length / (total_length / (len(lon) - 1))) + 1
        extended_lon_start = [lon[0] - delta_lon_start * (i * extension_length / num_points_start) for i in range(1, num_points_start + 1)][::-1]
        extended_lat_start = [lat[0] - delta_lat_start * (i * extension_length / num_points_start) for i in range(1, num_points_start + 1)][::-1]
        extended_Z_start = [Z[0]] * num_points_start
        extended_dist_start = [-i * extension_length / num_points_start for i in range(1, num_points_start + 1)][::-1]

        extended_lon = np.array(extended_lon_start + list(lon) + extended_lon_end)
        extended_lat = np.array(extended_lat_start + list(lat) + extended_lat_end)
        extended_Z = np.array(extended_Z_start + list(Z) + extended_Z_end)
        extended_distances = np.array(extended_dist_start + list(distances) + extended_dist_end)
        return extended_lon, extended_lat, extended_Z, extended_distances
    except Exception as e:
        st.error(f"Erreur lors de l'extension des coordonnées : {e}")
        return lon, lat, Z, distances

# --- Interpolation profil avec IDW et Spline ---
def interpolate_profile(distances, Z, lon, lat, method='idw', resolution=1000, extension_ratio=0.0):
    try:
        valid_mask = ~np.isnan(Z) & ~np.isnan(distances) & ~np.isnan(lon) & ~np.isnan(lat)
        if not np.any(valid_mask):
            st.error("Aucune donnée valide pour l'interpolation (Z, distances, lon, ou lat contiennent des NaN).")
            return None, None, None, None
        distances = distances[valid_mask]
        Z = Z[valid_mask]
        lon = lon[valid_mask]
        lat = lat[valid_mask]

        if not (len(distances) == len(Z) == len(lon) == len(lat)):
            st.error(f"Incohérence des longueurs après filtrage : distances={len(distances)}, Z={len(Z)}, lon={len(lon)}, lat={len(lat)}")
            return None, None, None, None

        total_length = distances[-1]
        extension_length = total_length * extension_ratio / 2
        interp_dist = np.linspace(distances.min() - extension_length, distances.max() + extension_length, resolution)
        interp_lon = np.interp(interp_dist, distances, lon)
        interp_lat = np.interp(interp_dist, distances, lat)

        if method == 'idw':
            interp_Z = np.zeros(resolution)
            p = 2
            for i, xd in enumerate(interp_dist):
                weights = 1 / (np.abs(distances - xd) ** p + 1e-10)
                weights /= weights.sum()
                interp_Z[i] = np.sum(weights * Z)
        elif method == 'spline':
            spline = UnivariateSpline(distances, Z, s=0.1, ext='extrapolate')
            interp_Z = spline(interp_dist)
        else:
            raise ValueError("Méthode d'interpolation non reconnue : choisissez 'idw' ou 'spline'.")
        
        if np.any(interp_Z > 0):
            st.warning(f"Attention : {np.sum(interp_Z > 0)} points interpolés ont des profondeurs positives.")
        return interp_dist, interp_Z, interp_lon, interp_lat
    except Exception as e:
        st.error(f"Erreur lors de l'interpolation du profil : {e}")
        return None, None, None, None

# --- Détection Foot of Slope (FoS) avec plusieurs résultats ---
def detect_foot_of_slope(interp_dist, interp_Z, slope_threshold=0.05):
    try:
        # Calcul de la première dérivée (pente)
        slope = np.gradient(interp_Z, interp_dist)
        # Calcul de la deuxième dérivée (changement de pente)
        curvature = np.gradient(slope, interp_dist)
        # Identifier les pics de courbure (changement maximal de gradient)
        curvature_peaks = np.where((curvature[:-1] * curvature[1:]) < 0)[0]  # Points où la courbure change de signe

        if len(curvature_peaks) == 0:
            st.warning("Aucun changement significatif de gradient détecté. Vérifiez les données ou augmentez la résolution.")
            return np.array([]), np.array([]), np.array([])

        # Filtrer les pics avec une pente supérieure au seuil et vérifier la transition
        foot_dist = []
        foot_Z = []
        foot_slopes = []
        for peak_idx in curvature_peaks:
            if abs(slope[peak_idx]) > slope_threshold:
                # Vérifier si la pente diminue après le pic (transition vers plaine abyssale)
                if peak_idx + 1 < len(slope) and abs(slope[peak_idx + 1]) < abs(slope[peak_idx]):
                    foot_dist.append(interp_dist[peak_idx])
                    foot_Z.append(interp_Z[peak_idx])
                    foot_slopes.append(slope[peak_idx])

        if not foot_dist:
            st.warning("Aucun FOS valide détecté : pas de transition vers une pente faible après les pics ou seuil trop élevé.")
            return np.array([]), np.array([]), np.array([])

        foot_dist = np.array(foot_dist)
        foot_Z = np.array(foot_Z)
        foot_slopes = np.array(foot_slopes)

        st.write(f"Nombre de points FOS détectés : {len(foot_dist)}")
        for i, (dist, z, s) in enumerate(zip(foot_dist, foot_Z, foot_slopes)):
            st.write(f"FOS {i+1} : Distance = {dist:.2f} m, Profondeur = {z:.2f} m, Pente = {s:.4f}")
            st.write(f"Validation : Pente avant = {slope[peak_idx - 1]:.4f}, Pente après = {slope[peak_idx + 1]:.4f}, Seuil utilisé = {slope_threshold:.2f}")

        return foot_dist, foot_Z, foot_slopes
    except Exception as e:
        st.error(f"Erreur lors de la détection des FoS : {e}")
        return np.array([]), np.array([]), np.array([])

# --- Calculer la distance géographique minimale d'un point CSV au profil ---
def calculate_min_distance_to_profile(csv_lon, csv_lat, profile_lon, profile_lat):
    try:
        if len(csv_lon) == 0 or len(csv_lat) == 0 or len(profile_lon) == 0 or len(profile_lat) == 0:
            st.warning("Tableaux CSV ou profil vides. Vérifiez les données d'entrée.")
            return float('inf'), -1, -1
        
        min_distances = []
        closest_profile_indices = []
        for i, (clon, clat) in enumerate(zip(csv_lon, csv_lat)):
            distances = [geodesic((clat, clon), (plat, plon)).meters for plat, plon in zip(profile_lat, profile_lon)]
            if not distances:
                min_distances.append(float('inf'))
                closest_profile_indices.append(-1)
                continue
            min_dist = min(distances)
            min_distances.append(min_dist)
            closest_profile_indices.append(np.argmin(distances))
        
        if not min_distances or all(d == float('inf') for d in min_distances):
            st.warning("Aucune distance valide calculée. Vérifiez les données CSV ou le profil.")
            return float('inf'), -1, -1
        
        min_distance_idx = np.argmin(min_distances)
        min_distance = min_distances[min_distance_idx]
        closest_csv_index = min_distance_idx
        closest_profile_index = closest_profile_indices[min_distance_idx]
        return min_distance, closest_csv_index, closest_profile_index
    except Exception as e:
        st.error(f"Erreur lors du calcul de la distance minimale au profil : {e}")
        return float('inf'), -1, -1

# --- Validation des données CSV ---
def validate_csv_data(fos_data, fos_column):
    required_columns = ['Latitude', 'Longitude', fos_column, 'Water_depth']
    missing_columns = [col for col in required_columns if col not in fos_data.columns]
    if missing_columns:
        st.error(f"Colonnes manquantes dans le CSV : {missing_columns}. Colonnes attendues : {required_columns}")
        return None, None, None, None
    
    try:
        csv_lat = pd.to_numeric(fos_data['Latitude'], errors='coerce')
        csv_lon = pd.to_numeric(fos_data['Longitude'], errors='coerce')
        csv_depth = pd.to_numeric(fos_data['Water_depth'], errors='coerce')
        csv_fos = fos_data[fos_column]
        
        invalid_lat = fos_data['Latitude'][csv_lat.isna()]
        invalid_lon = fos_data['Longitude'][csv_lon.isna()]
        invalid_depth = fos_data['Water_depth'][csv_depth.isna()]
        
        if len(invalid_lat) > 0:
            st.warning(f"Valeurs non numériques dans 'Latitude' (premières 5) : {invalid_lat.head().tolist()}")
            st.info("Solution : Remplacez les valeurs non numériques (ex. 'N/A', '') par des nombres valides ou supprimez les lignes.")
        if len(invalid_lon) > 0:
            st.warning(f"Valeurs non numériques dans 'Longitude' (premières 5) : {invalid_lon.head().tolist()}")
            st.info("Solution : Remplacez les valeurs non numériques (ex. 'N/A', '') par des nombres valides ou supprimez les lignes.")
        if len(invalid_depth) > 0:
            st.warning(f"Valeurs non numériques dans 'Water_depth' (premières 5) : {invalid_depth.head().tolist()}")
            st.info("Solution : Remplacez les valeurs non numériques (ex. 'N/A', '') par des nombres valides ou supprimez les lignes.")
        
        valid_mask = ~(csv_lat.isna() | csv_lon.isna() | csv_depth.isna())
        if not valid_mask.any():
            st.error("Aucune donnée valide dans les colonnes Latitude, Longitude ou Water_depth après conversion.")
            return None, None, None, None
        
        st.write(f"Nombre de lignes valides après validation : {sum(valid_mask)}")
        return csv_lat[valid_mask].values, csv_lon[valid_mask].values, csv_fos[valid_mask].values, csv_depth[valid_mask].values
    except Exception as e:
        st.error(f"Erreur lors de la validation des données CSV : {e}")
        return None, None, None, None

# --- Identifier le point FOS du CSV traversé par le profil ---
def find_fos_on_profile(interp_dist, interp_Z, fos_data, fos_column, interp_lon, interp_lat):
    if interp_dist is None or interp_Z is None or interp_lon is None or interp_lat is None:
        st.error("Données d'interpolation non valides. Vérifiez les données VTK.")
        return None, None, None, None, None
    
    csv_lat, csv_lon, csv_fos, csv_depth = validate_csv_data(fos_data, fos_column)
    if csv_lat is None or len(csv_lat) == 0:
        st.error("Aucune donnée CSV valide après validation. Vérifiez le fichier CSV.")
        return None, None, None, None, None
    
    try:
        min_distance, closest_csv_index, closest_profile_index = calculate_min_distance_to_profile(csv_lon, csv_lat, interp_lon, interp_lat)
        if closest_csv_index == -1 or closest_profile_index == -1 or closest_csv_index >= len(csv_fos):
            st.warning("Aucun point FOS CSV trouvé à proximité du profil ou index invalide.")
            return None, None, None, None, None
        
        closest_fos_id = csv_fos[closest_csv_index]
        closest_csv_lon = csv_lon[closest_csv_index]
        closest_csv_lat = csv_lat[closest_csv_index]
        closest_csv_depth = csv_depth[closest_csv_index]
        closest_fos_dist = interp_dist[closest_profile_index]
        
        st.write(f"Point FOS CSV traversé par le profil : FOS_ID = {closest_fos_id}, Distance au profil = {min_distance:.2f} mètres, Profondeur = {closest_csv_depth:.2f} m")
        
        return closest_fos_id, closest_csv_lon, closest_csv_lat, closest_csv_depth, closest_fos_dist
    except Exception as e:
        st.error(f"Erreur lors de l'identification du point FOS CSV : {e}")
        return None, None, None, None, None

# --- Extraction des valeurs FOS depuis GeoTIFF ---
def extract_fos_from_geotiff(fos_tiff_data, fos_tiff_transform, lon, lat):
    fos_values = []
    for lo, la in zip(lon, lat):
        try:
            row, col = rasterio.transform.rowcol(fos_tiff_transform, lo, la)
            if 0 <= row < fos_tiff_data.shape[0] and 0 <= col < fos_tiff_data.shape[1]:
                fos_values.append(fos_tiff_data[int(row), int(col)])
            else:
                fos_values.append(np.nan)
        except Exception as e:
            st.warning(f"Erreur lors de l'extraction FOS pour ({lo}, {la}) : {e}")
            fos_values.append(np.nan)
    return np.array(fos_values)

# --- Convertir les données GeoTIFF en image pour l'affichage ---
def create_image_overlay(data, bounds, cmap_name="magma"):
    try:
        max_size = 300
        if data.shape[0] > max_size or data.shape[1] > max_size:
            scale = max_size / max(data.shape)
            data = zoom(data, scale, order=1)
        
        norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        cmap = cm.get_cmap(cmap_name)
        img_data = cmap(norm(data))[:, :, :3]
        img_data = (img_data * 255).astype(np.uint8)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            img_path = tmp_file.name

        fig, ax = plt.subplots()
        ax.imshow(img_data, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        ax.axis("off")
        plt.savefig(img_path, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Échec de la création du fichier : {img_path}")

        with open(img_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        os.remove(img_path)
        return encoded
    except Exception as e:
        st.error(f"Erreur lors de la création de l'image GeoTIFF : {str(e)}")
        return None

# --- Carte Folium avec annotations FoS, points CSV et profil VTK ---
def plot_foot_of_slope_map(lon, lat, foot_lon, foot_lat, foot_Z, fos_data=None, fos_column='FOS_ID', fos_tiff_data=None, fos_tiff_bounds=None, display_geotiff=True):
    if lon is None or lat is None:
        st.error("Données VTK non valides pour l'affichage de la carte.")
        return None
    
    m = folium.Map(location=[np.mean(lat), np.mean(lon)], zoom_start=10, tiles="cartodbpositron")
    
    if display_geotiff and fos_tiff_data is not None and fos_tiff_bounds is not None:
        encoded_fos_image = create_image_overlay(fos_tiff_data, fos_tiff_bounds)
        if encoded_fos_image:
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{encoded_fos_image}",
                bounds=[[fos_tiff_bounds.bottom, fos_tiff_bounds.left], [fos_tiff_bounds.top, fos_tiff_bounds.right]],
                opacity=0.5,
                name="FOS Layer"
            ).add_to(m)
            folium.LayerControl().add_to(m)

    profile_points = list(zip(lat, lon))
    folium.PolyLine(
        locations=profile_points,
        color='black',
        weight=2,
        opacity=0.8,
        popup="Profil bathymétrique VTK"
    ).add_to(m)

    for lo, la, depth in zip(foot_lon, foot_lat, foot_Z):
        folium.CircleMarker(
            location=[la, lo],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=f"Profondeur (calculée) : {depth:.2f} m"
        ).add_to(m)
    
    if fos_data is not None:
        csv_lat, csv_lon, csv_fos, csv_depth = validate_csv_data(fos_data, fos_column)
        if csv_lat is not None:
            for la, lo, fos_id, depth in zip(csv_lat, csv_lon, csv_fos, csv_depth):
                folium.CircleMarker(
                    location=[la, lo],
                    radius=3,
                    color='blue',
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"FOS_ID (CSV) : {fos_id}, Profondeur (CSV) : {depth:.2f} m"
                ).add_to(m)
        else:
            st.warning("Aucun point FOS du CSV n'a pu être affiché en raison de données non valides.")

    return m

# --- Fonction export CSV FoS ---
def export_foot_of_slope_csv(foot_lon, foot_lat, foot_Z, foot_fos=None):
    data = {
        "Longitude": foot_lon,
        "Latitude": foot_lat,
        "Profondeur": foot_Z
    }
    if foot_fos is not None:
        data["FOS_ID"] = foot_fos
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# --- Application Streamlit ---
def main():
    st.title("🛠️ Visualisation Foot of Slope depuis un fichier VTK avec analyse FOS")
    st.info("Profondeurs doivent être en mètres, négatives (sous le niveau de la mer). CSV doit avoir des colonnes numériques pour Latitude, Longitude, Water_depth. Ajustez le seuil de pente pour affiner la détection de multiples FOS.")

    # Charger le fichier VTK
    uploaded_file = st.file_uploader("Choisissez un fichier VTK", type=["vtk"])
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".vtk") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            mesh = pv.read(tmp_path)
            points = extract_coords_from_mesh(mesh)
            if points is None:
                return
            
            st.write(f"Nombre de points extraits du VTK : {len(points)}")
            lon, lat, Z = convert_utm_to_latlon(points)
            if lon is None:
                return

            st.write(f"Longitude min/max : {np.min(lon):.4f}, {np.max(lon):.4f}")
            st.write(f"Latitude min/max : {np.min(lat):.4f}, {np.max(lat):.4f}")
            st.write(f"Profondeur min/max : {np.min(Z):.2f}, {np.max(Z):.2f} m")
            st.write(f"Longueur des tableaux lon/lat/Z : {len(lon)}, {len(lat)}, {len(Z)}")

            # Charger le fichier CSV pour FOS
            st.header("Charger les données FOS (CSV)")
            st.info("Exemple de CSV : FOS_PROT05MV_S,-4.123,39.456,-200.0")
            uploaded_csv = st.file_uploader("Charger un fichier CSV avec les données FOS", type=["csv"])
            fos_data = None
            fos_column = st.text_input("Nom de la colonne FOS dans le CSV", value="FOS_ID")
            if uploaded_csv is not None:
                try:
                    fos_data = pd.read_csv(uploaded_csv)
                    st.write("Données FOS chargées avec succès !")
                    st.write("Colonnes disponibles dans le CSV :", list(fos_data.columns))
                    st.write("Aperçu des données CSV (premières 5 lignes) :")
                    st.dataframe(fos_data.head())
                    if fos_column in fos_data.columns:
                        st.write(f"Premières valeurs de la colonne '{fos_column}' :", fos_data[fos_column].head().tolist())
                    else:
                        st.warning(f"La colonne '{fos_column}' n'est pas présente dans le CSV.")
                    if 'Water_depth' in fos_data.columns:
                        st.write(f"Premières valeurs de la colonne 'Water_depth' :", fos_data['Water_depth'].head().tolist())
                    else:
                        st.warning("La colonne 'Water_depth' n'est pas présente dans le CSV.")
                except Exception as e:
                    st.error(f"Erreur lors du chargement du CSV : {e}")
                    fos_data = None

            # Charger le GeoTIFF pour le fond de carte FOS
            st.header("Charger le GeoTIFF pour le fond de carte FOS")
            display_geotiff = st.checkbox("Afficher le GeoTIFF dans la carte Folium", value=True)
            uploaded_fos_tiff = st.file_uploader("Charger un fichier GeoTIFF FOS", type=["tif", "tiff"])
            fos_tiff_data = None
            fos_tiff_bounds = None
            fos_tiff_transform = None
            if uploaded_fos_tiff is not None:
                try:
                    with rasterio.open(uploaded_fos_tiff) as dataset:
                        fos_tiff_data = dataset.read(1)
                        fos_tiff_transform = dataset.transform
                        fos_tiff_bounds = dataset.bounds
                        fos_tiff_crs = dataset.crs
                    st.write("Fond de carte FOS chargé avec succès !")
                    st.write(f"Dimensions du GeoTIFF : {fos_tiff_data.shape}")
                except Exception as e:
                    st.error(f"Erreur lors du chargement du GeoTIFF FOS : {e}")
                    fos_tiff_data = None

            # Ajouter un contrôle pour l'extension du profil
            st.header("Paramètres du profil bathymétrique")
            extension_percentage = st.slider(
                "Extension du profil (% de la longueur totale)",
                0.0, 50.0, 0.0, step=5.0,
                help="Étendre le profil au-delà des points VTK (symétrique aux deux extrémités)."
            )
            extension_ratio = extension_percentage / 100.0

            # Ajouter un contrôle pour la résolution
            resolution = st.slider("Résolution interpolation", 100, 2000, 1000, step=100)

            # Ajouter un contrôle pour le seuil de pente
            slope_threshold = st.slider(
                "Seuil de pente pour FoS",
                min_value=0.01, max_value=0.2, value=0.05, step=0.01,
                help="Ajustez le seuil de pente pour affiner la détection des multiples FOS (recommandation CLCS : justifier le choix selon le profil)."
            )

            # Étendre les coordonnées, profondeurs et distances
            if extension_ratio > 0:
                lon, lat, Z, distances = extend_coordinates(lon, lat, Z, compute_cumulative_distances(lon, lat), extension_ratio)
                if distances is None:
                    return
            else:
                distances = compute_cumulative_distances(lon, lat)
                if distances is None:
                    return

            st.write(f"Longueur après extension - lon/lat/Z/distances : {len(lon)}, {len(lat)}, {len(Z)}, {len(distances)}")

            method = st.selectbox("Méthode d'interpolation", ['Gridding IDW', 'Spline'], index=0)
            method_map = {'Gridding IDW': 'idw', 'Spline': 'spline'}
            interp_dist, interp_Z, interp_lon, interp_lat = interpolate_profile(distances, Z, lon, lat, method=method_map[method], resolution=resolution, extension_ratio=extension_ratio)
            if interp_dist is None:
                return

            st.write(f"Longueur après interpolation - interp_dist/interp_lon/interp_lat : {len(interp_dist)}, {len(interp_lon)}, {len(interp_lat)}")

            foot_dist, foot_Z, foot_slopes = detect_foot_of_slope(interp_dist, interp_Z, slope_threshold)

            foot_lon = np.array([])
            foot_lat = np.array([])
            if len(foot_dist) > 0:
                foot_lon = np.interp(foot_dist, interp_dist, interp_lon)
                foot_lat = np.interp(foot_dist, interp_dist, interp_lat)

            foot_fos = None
            if fos_tiff_data is not None and len(foot_lon) > 0:
                foot_fos = extract_fos_from_geotiff(fos_tiff_data, fos_tiff_transform, foot_lon, foot_lat)
                st.write("Valeurs FOS extraites du GeoTIFF (premières 5) :", foot_fos[:5].tolist())

            csv_fos_dist = None
            csv_fos_depth = None
            csv_fos_id = None
            if fos_data is not None:
                closest_fos_id, closest_csv_lon, closest_csv_lat, csv_fos_depth, csv_fos_dist = find_fos_on_profile(interp_dist, interp_Z, fos_data, fos_column, interp_lon, interp_lat)
                if closest_fos_id is not None:
                    csv_fos_id = closest_fos_id
                else:
                    st.warning("Impossible d'identifier le point FOS CSV : vérifiez les données ou le nom de la colonne.")
            else:
                st.warning("Données FOS CSV manquantes pour l'affichage du point FOS. Veuillez charger un fichier CSV valide.")

            # Ajouter un contrôle pour l'échelle de profondeur
            st.header("Échelle de profondeur du profil bathymétrique")
            st.info("Le facteur d'échelle de profondeur modifie la forme du profil (compression ou étirement vertical). Une valeur < 1 aplatit la courbe, une valeur > 1 l'exagère.")
            depth_scale_factor = st.slider(
                "Facteur d'échelle de profondeur",
                min_value=0.1, max_value=1.0, value=1.0, step=0.01,
                help="Ajustez pour compresser (<1) ou étirer (>1) le profil bathymétrique verticalement."
            )

            # Option pour activer les curseurs de plage de profondeur
            use_depth_range = st.checkbox("Utiliser une plage de profondeur personnalisée", value=False)
            all_depths = np.concatenate([Z, interp_Z]) if interp_Z is not None else Z
            default_y_min = float(np.floor(np.min(all_depths) / 100) * 100)
            default_y_max = float(np.ceil(np.max(all_depths) / 100) * 100)
            if use_depth_range:
                y_min = st.slider("Profondeur minimale (m)", 
                                min_value=float(default_y_min - 500), 
                                max_value=float(default_y_max), 
                                value=float(default_y_min), 
                                step=10.0,
                                help="Ajustez la profondeur minimale affichée (négative, sous le niveau de la mer).")
                y_max = st.slider("Profondeur maximale (m)", 
                                min_value=float(default_y_min), 
                                max_value=float(default_y_max + 500), 
                                value=float(default_y_max), 
                                step=10.0,
                                help="Ajustez la profondeur maximale affichée (négative, sous le niveau de la mer).")
                if y_min > y_max:
                    st.error("Erreur : La profondeur minimale ne peut pas être supérieure à la profondeur maximale.")
                    y_min, y_max = y_max, y_min
            else:
                y_min = default_y_min
                y_max = default_y_max

            # Appliquer l'échelle de profondeur aux données
            Z_scaled = Z * depth_scale_factor
            interp_Z_scaled = interp_Z * depth_scale_factor if interp_Z is not None else None
            foot_Z_scaled = foot_Z * depth_scale_factor if len(foot_Z) > 0 else np.array([])
            csv_fos_depth_scaled = csv_fos_depth * depth_scale_factor if csv_fos_depth is not None else None

            # Calculer l'orientation
            def calculate_bearing(start, end):
                try:
                    lon1, lat1 = math.radians(start[0]), math.radians(start[1])
                    lon2, lat2 = math.radians(end[0]), math.radians(end[1])
                    delta_lon = lon2 - lon1
                    x = math.sin(delta_lon) * math.cos(lat2)
                    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
                    bearing = math.atan2(x, y)
                    bearing = math.degrees(bearing)
                    bearing = (bearing + 360) % 360
                    bearing = (bearing + 180) % 360  # Ajuster pour corriger l'inversion
                    return bearing
                except Exception as e:
                    st.error(f"Erreur dans le calcul du bearing : {str(e)}")
                    return 0.0

            def bearing_to_cardinal(bearing):
                try:
                    directions = [
                        "NORD", "NORD-NORD-EST", "NORD-EST", "EST-NORD-EST",
                        "EST", "EST-SUD-EST", "SUD-EST", "SUD-SUD-EST",
                        "SUD", "SUD-SUD-OUEST", "SUD-OUEST", "OUEST-SUD-OUEST",
                        "OUEST", "OUEST-NORD-OUEST", "NORD-OUEST", "NORD-NORD-OUEST"
                    ]
                    index = int((bearing + 11.25) / 22.5) % 16
                    return directions[index]
                except Exception as e:
                    st.error(f"Erreur dans la conversion en cardinale : {str(e)}")
                    return "Inconnu"

            # Vérification des données avant le calcul de l'orientation
            if len(lon) >= 2 and len(lat) >= 2 and not np.any(np.isnan(lon)) and not np.any(np.isnan(lat)):
                try:
                    start_bearing = calculate_bearing((lon[0], lat[0]), (lon[-1], lat[-1]))
                    end_bearing = calculate_bearing((lon[-1], lat[-1]), (lon[0], lat[0]))
                    start_direction = bearing_to_cardinal(start_bearing)
                    end_direction = bearing_to_cardinal(end_bearing)
                    st.write(f"Débogage : Orientation début = {start_direction}, fin = {end_direction}")
                except Exception as e:
                    st.error(f"Erreur lors du calcul des orientations : {str(e)}")
                    start_direction, end_direction = "Inconnu", "Inconnu"
            else:
                st.warning("Données insuffisantes ou invalides pour calculer l'orientation (moins de 2 points ou NaN détecté).")
                start_direction, end_direction = "Inconnu", "Inconnu"

            # Affichage profil avec Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=distances, y=Z_scaled, mode='markers', name='Points originaux',
                marker=dict(color='black', size=5)
            ))
            fig.add_trace(go.Scatter(
                x=interp_dist, y=interp_Z_scaled, mode='lines', name='Profil interpolé',
                line=dict(color="#87CEEB", width=3, dash="solid"),
                fill="tozeroy",
                fillcolor="rgba(135, 206, 235, 0.3)"
            ))
            if len(foot_dist) > 0:
                fig.add_trace(go.Scatter(
                    x=foot_dist, y=foot_Z_scaled, mode='markers', name='FoS (changement max gradient)',
                    marker=dict(color='red', size=8),
                    hovertemplate='Distance: %{x:.2f} m<br>Profondeur (mise à l\'échelle): %{y:.2f} m'
                ))
            if csv_fos_dist is not None and csv_fos_depth_scaled is not None:
                fig.add_trace(go.Scatter(
                    x=[csv_fos_dist], y=[csv_fos_depth_scaled], mode='markers', name=f'FOS CSV ({csv_fos_id})',
                    marker=dict(color='green', size=10, symbol='diamond'),
                    hovertemplate=f'FOS_ID: {csv_fos_id}<br>Distance: %{{x:.2f}} m<br>Profondeur (mise à l\'échelle): %{{y:.2f}} m'
                ))

            # Ajouter les annotations pour l'orientation avec un décalage dynamique
            annotations = []
            if len(distances) > 0 and interp_Z_scaled is not None:
                depth_range = max(interp_Z_scaled) - min(interp_Z_scaled)
                offset = depth_range * 0.15 if depth_range != 0 else 150
                if not np.isnan(distances[0]) and not np.isnan(Z_scaled[0]):
                    annotations.append(
                        dict(
                            x=distances[0],
                            y=Z_scaled[0] + offset,
                            xref="x",
                            yref="y",
                            text=start_direction if start_direction != "Inconnu" else "Non défini",
                            showarrow=False,
                            xanchor="center",
                            yanchor="bottom",
                            font=dict(size=12, color="#4682B4")
                        )
                    )
                if not np.isnan(distances[-1]) and not np.isnan(Z_scaled[-1]):
                    annotations.append(
                        dict(
                            x=distances[-1],
                            y=Z_scaled[-1] + offset,
                            xref="x",
                            yref="y",
                            text=end_direction if end_direction != "Inconnu" else "Non défini",
                            showarrow=False,
                            xanchor="center",
                            yanchor="bottom",
                            font=dict(size=12, color="#4682B4")
                        )
                    )

            fig.update_layout(
                title="Profil bathymétrique avec FoS détectés et point FOS CSV (zoomez pour voir les pentes)",
                xaxis_title="Distance cumulée (m)",
                yaxis_title="Profondeur (m, mise à l'échelle)",
                yaxis=dict(range=[y_max, y_min], autorange=None),
                showlegend=True,
                width=700,
                height=500,
                plot_bgcolor="rgba(240, 248, 255, 0.8)",
                paper_bgcolor="rgba(240, 248, 255, 0.8)",
                font=dict(color="#333333"),
                annotations=annotations
            )
            st.plotly_chart(fig, use_container_width=True)

            # Carte
            st.subheader("Localisation des FoS (rouges : changement max gradient, bleus : CSV, noir : profil VTK)")
            folium_map = plot_foot_of_slope_map(interp_lon, interp_lat, foot_lon, foot_lat, foot_Z, fos_data, fos_column, fos_tiff_data, fos_tiff_bounds, display_geotiff)
            if folium_map is not None:
                st_folium(folium_map, width=700, height=500)

            # Export CSV dynamique
            st.subheader("Export des points Foot of Slope (FoS, changement max gradient)")
            csv_data = export_foot_of_slope_csv(foot_lon, foot_lat, foot_Z, foot_fos)
            st.download_button(
                label="Télécharger les FoS (changement max gradient) au format CSV",
                data=csv_data,
                file_name="foot_of_slope_max_change.csv",
                mime="text/csv"
            )

        finally:
            if 'tmp_path' in locals():
                os.remove(tmp_path)

if __name__ == "__main__":
    main()