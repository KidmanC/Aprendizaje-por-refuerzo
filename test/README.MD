# Setup
1. Instalar Banana Kong en el emulador BlueStacks (Android 11)
[Descargar](https://cloud.bluestacks.com/api/getdownloadnow?platform=win&win_version=11&mac_version=&client_uuid=6d350092-4fe0-486f-a912-9f47b7bb530b&app_pkg=nxt_rvc64&platform_cloud=&preferred_lang=es&utm_source=&utm_medium=&gaCookie=GA1.1.622681839.1771804238&gclid=&clickid=&msclkid=&affiliateId=&offerId=&transaction_id=&aff_sub=&referrer=https%253A%252F%252Fwww.google.com%252F&first_landing_page=https%253A%252F%252Fwww.bluestacks.com%252Fes%252Findex.html&download_page_referrer=https%3A%2F%2Fwww.bluestacks.com%2Fdownload.html&utm_campaign=homepage-dl-button-es&user_id=experiment_variant&exit_utm_campaign=nxt-rvc64-button-download-page-es&incompatible=false&bluestacks_version=bs5&device_memory=8&device_cpu_cores=12&extra_data=%7B%22deviceDetails%22%3A%22windows%22%2C%22renderer%22%3A%22ANGLE%20(AMD%2C%20AMD%20Radeon%20RX%209070%20XT%20(0x00007550)%20Direct3D11%20vs_5_0%20ps_5_0%2C%20D3D11)%22%7D)
2. Resolucion: 960x540
3. Crear entorno virtual de Python:
```bash
python -m venv .venv
```
4. Activar entorno virtual:
```bash
python .venv/Scripts/activate
```
5. Instalar dependecias (Desde la carpeta test)
```bash
pip install -r requirements
```