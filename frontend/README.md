# IIoT Motor Monitoring Dashboard

React dashboard connected to the Industrial IoT backend. It polls live MQTT status every second and Firebase history every 30 seconds. Live sensor values are shown only after their MQTT record is successfully written to Firebase; older Firebase records are explicitly labelled historical. A 15-second default freshness timeout marks stale data as not updating.

Motor Intelligence contains XGBoost class prediction history and
Motor Health & Alerts. Prediction probabilities are identified as uncalibrated
model probabilities; AI predictions are not presented as confirmed failures.
Unavailable fields render as “No Data”.

## Run the application

Start the backend from the project root (see [backend setup](../backend/README.md)):

```sh
. .venv/bin/activate
python -m backend.main
```

In a second terminal:

```sh
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). Vite proxies `/health`
and `/api/*` to `http://127.0.0.1:8000`. Set `VITE_BACKEND_URL` before starting
Vite if the backend runs at a different address.

For a separately hosted frontend, set `VITE_API_BASE_URL` to the backend's
origin when building, and add the frontend origin to `CORS_ORIGINS` in
`backend/.env`.

## Available scripts

- `npm run dev` — start the development server.
- `npm run build` — create a production build in `dist/`.
- `npm run preview` — preview the production build.
- `npm run lint` — check the frontend source.
- `npm test` — run dashboard data-mapping tests.

## Project structure

- `src/components/` — reusable dashboard and navigation components.
- `src/pages/` — dashboard screens.
- `src/api/client.js` — health and backend API requests.
- `src/api/dashboard.js` — backend response mapping for dashboard components.

- `src/pages/AIAnomalyDetection/` — XGBoost prediction history.
- `src/pages/MotorHealthAlerts/` — AI alerts and sensor-threshold status.
- `src/data/` — dashboard descriptions.
- `src/utils/` — shared formatting and status helpers.
- `src/assets/` — frontend images and assets.
- `public/` — static files served as-is.

All application source and frontend tooling are kept in this directory.
