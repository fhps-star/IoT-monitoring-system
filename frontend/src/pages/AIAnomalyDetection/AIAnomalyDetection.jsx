import { useEffect, useState, useCallback } from "react";
import { Cpu, Check, AlertTriangle, RefreshCw } from "lucide-react";
import { EmptyState, Panel, TableToolbar } from "../../components/shared.jsx";
import {
  fetchModelEvaluation,
  fetchLiveComparison,
} from "../../api/client.js";
import "./AIAnomalyDetection.css";

const classTone = (label) => (label === "NORMAL" ? "normal" : "anomaly");

function formatPercent(val) {
  if (typeof val !== "number" || !Number.isFinite(val)) return "N/A";
  return `${(val * 100).toFixed(2)}%`;
}

function formatLatency(val) {
  if (typeof val !== "number" || !Number.isFinite(val)) return "N/A";
  return `${val.toFixed(2)} ms`;
}

function formatCount(val) {
  if (typeof val !== "number" || !Number.isFinite(val)) return "0";
  return val.toLocaleString();
}

function getBetterOffline(rfVal, xgbVal, lowerIsBetter = false) {
  if (
    typeof rfVal !== "number" ||
    !Number.isFinite(rfVal) ||
    typeof xgbVal !== "number" ||
    !Number.isFinite(xgbVal)
  ) {
    return null;
  }
  if (Math.abs(rfVal - xgbVal) < 1e-5) return "tie";
  if (lowerIsBetter) {
    return rfVal < xgbVal ? "rf" : "xgb";
  }
  return rfVal > xgbVal ? "rf" : "xgb";
}

export function AIAnomalyDetection({ predictions = [], modelStatus = "unavailable" }) {
  const [offlineState, setOfflineState] = useState({
    data: null,
    loading: true,
    error: null,
  });

  const [liveCompState, setLiveCompState] = useState({
    data: null,
    loading: true,
    error: null,
  });

  const loadOffline = useCallback(async (signal) => {
    try {
      setOfflineState((prev) => ({ ...prev, loading: true, error: null }));
      const data = await fetchModelEvaluation(signal);
      setOfflineState({ data, loading: false, error: null });
    } catch (err) {
      if (err.name !== "AbortError") {
        setOfflineState({ data: null, loading: false, error: err.message });
      }
    }
  }, []);

  const loadLive = useCallback(async (signal) => {
    try {
      const comp = await fetchLiveComparison(signal);
      setLiveCompState({ data: comp, loading: false, error: null });
    } catch (err) {
      if (err.name !== "AbortError") {
        setLiveCompState((prev) => ({ ...prev, loading: false, error: err.message }));
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadOffline(controller.signal);
    loadLive(controller.signal);

    const interval = setInterval(() => {
      loadLive(controller.signal);
    }, 2000);

    return () => {
      controller.abort();
      clearInterval(interval);
    };
  }, [loadOffline, loadLive]);

  // Offline metrics extraction
  const rfOffline = offlineState.data?.random_forest || {};
  const xgbOffline = offlineState.data?.xgboost || {};

  // Live comparison data extraction
  const liveComp = liveCompState.data || {};
  const rfLive = liveComp.random_forest || {};
  const xgbLive = liveComp.xgboost || {};
  const rfLatestPred = rfLive.latest_prediction;
  const xgbLatestPred = xgbLive.latest_prediction;

  let modelsAgree = null;
  if (rfLatestPred && xgbLatestPred) {
    modelsAgree = rfLatestPred === xgbLatestPred;
  }

  return (
    <div className="page-stack ai-prediction-page">
      {/* 2. Selected ML Model status */}
      <Panel className="ai-model-status-panel">
        <div className="ai-model-icon">
          <Cpu size={18} />
        </div>
        <div className="ai-model-copy">
          <strong>Selected ML Models: Random Forest &amp; XGBoost</strong>
          <span>
            Model status: {modelStatus === "trained" ? "Ready" : "Unavailable"} · Multiclass Anomaly Classification
          </span>
        </div>
      </Panel>

      {/* 3. Prediction History */}
      <Panel className="ai-prediction-panel">
        <TableToolbar title="Prediction history" count={predictions.length} />
        {predictions.length ? (
          <div className="table-scroll ai-prediction-scroll">
            <table className="data-table ai-prediction-table">
              <thead>
                <tr>
                  <th>Predicted class</th>
                  <th>Model probability</th>
                  <th>Prediction time</th>
                  <th>Telemetry record</th>
                  <th>State</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <span className={`ai-class-pill ${classTone(item.predicted_class)}`}>
                        {item.predicted_class}
                      </span>
                    </td>
                    <td>
                      {typeof item.confidence === "number" && Number.isFinite(item.confidence)
                        ? `${(item.confidence * 100).toFixed(1)}% · uncalibrated`
                        : "No Data"}
                    </td>
                    <td>
                      {item.prediction_timestamp
                        ? new Date(item.prediction_timestamp).toLocaleString()
                        : "No Data"}
                    </td>
                    <td className="ai-telemetry-id">{item.telemetry_record_id ?? "No Data"}</td>
                    <td>{item.historical ? "Historical" : "Latest"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title={
              modelStatus === "trained"
                ? "No XGBoost predictions yet"
                : "XGBoost model unavailable"
            }
            detail={
              modelStatus === "trained"
                ? "A prediction appears after complete, valid sensor telemetry is stored."
                : "Check the saved model artifacts and backend logs."
            }
          />
        )}
      </Panel>

      {/* 4. Offline Model Evaluation */}
      <Panel className="comparison-table-panel">
        <div className="comparison-panel-heading">
          <div className="comparison-panel-title">
            <h2>Offline Model Evaluation</h2>
            <span className="comparison-panel-subtitle">
              Chronological holdout benchmark (500 records)
            </span>
          </div>
          <div className="comparison-header-actions">
            <span className="comparison-badge">Trained Evaluation</span>
          </div>
        </div>

        {offlineState.error ? (
          <div className="comparison-error-state">
            <AlertTriangle size={16} />
            <span>Failed to load offline model evaluation: {offlineState.error}</span>
            <button
              type="button"
              className="retry-button"
              onClick={() => {
                const c = new AbortController();
                loadOffline(c.signal);
              }}
            >
              <RefreshCw size={12} /> Retry
            </button>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th className="th-metric">Evaluation Metrics</th>
                  <th className="th-model">Random Forest</th>
                  <th className="th-model">XGBoost</th>
                </tr>
              </thead>
              <tbody>
                {[
                  {
                    metric: "Accuracy (%)",
                    rf: rfOffline.accuracy,
                    xgb: xgbOffline.accuracy,
                    lowerIsBetter: false,
                    format: formatPercent,
                  },
                  {
                    metric: "Precision (%)",
                    rf: rfOffline.precision,
                    xgb: xgbOffline.precision,
                    lowerIsBetter: false,
                    format: formatPercent,
                  },
                  {
                    metric: "Recall (%)",
                    rf: rfOffline.recall,
                    xgb: xgbOffline.recall,
                    lowerIsBetter: false,
                    format: formatPercent,
                  },
                  {
                    metric: "F1-Score (%)",
                    rf: rfOffline.f1_score,
                    xgb: xgbOffline.f1_score,
                    lowerIsBetter: false,
                    format: formatPercent,
                  },
                  {
                    metric: "Inference Time (500 Records)",
                    rf: rfOffline.inference_time_500_records,
                    xgb: xgbOffline.inference_time_500_records,
                    lowerIsBetter: true,
                    format: formatLatency,
                  },
                ].map(({ metric, rf, xgb, lowerIsBetter, format }) => {
                  const better = getBetterOffline(rf, xgb, lowerIsBetter);
                  return (
                    <tr key={metric}>
                      <td className="cell-metric">{metric}</td>
                      <td className={`cell-value ${better === "rf" ? "better-metric" : ""}`}>
                        <span className="val-text">{format(rf)}</span>
                        {better === "rf" && <span className="better-badge">Better</span>}
                      </td>
                      <td className={`cell-value ${better === "xgb" ? "better-metric" : ""}`}>
                        <span className="val-text">{format(xgb)}</span>
                        {better === "xgb" && <span className="better-badge">Better</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {/* 5. Live Model Performance */}
      <Panel className="comparison-table-panel">
        <div className="comparison-panel-heading">
          <div className="comparison-panel-title">
            <h2>Live Model Performance</h2>
            <span className="live-indicator-pill">
              <span className="live-dot" /> LIVE
            </span>
            {liveComp.last_updated && (
              <span className="comparison-timestamp">
                Latest: {new Date(liveComp.last_updated).toLocaleTimeString()}
              </span>
            )}
          </div>
          <div className="comparison-header-actions">
            {modelsAgree === true && (
              <span className="agreement-pill agree">
                <Check size={12} /> Models Agree
              </span>
            )}
            {modelsAgree === false && (
              <span className="agreement-pill disagree">
                <AlertTriangle size={12} /> Models Disagree
              </span>
            )}
            {modelsAgree === null && (
              <span className="agreement-pill waiting">Awaiting Telemetry</span>
            )}
          </div>
        </div>

        {liveCompState.error ? (
          <div className="comparison-error-state">
            <AlertTriangle size={16} />
            <span>Connection error updating live performance: {liveCompState.error}</span>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th className="th-metric">Live Performance Metrics</th>
                  <th className="th-model">Random Forest</th>
                  <th className="th-model">XGBoost</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="cell-metric">Latest Prediction</td>
                  <td className="cell-value">
                    {rfLatestPred ? (
                      <span className={`ai-class-pill ${classTone(rfLatestPred)}`}>
                        {rfLatestPred}
                      </span>
                    ) : (
                      <span className="na-text">N/A</span>
                    )}
                  </td>
                  <td className="cell-value">
                    {xgbLatestPred ? (
                      <span className={`ai-class-pill ${classTone(xgbLatestPred)}`}>
                        {xgbLatestPred}
                      </span>
                    ) : (
                      <span className="na-text">N/A</span>
                    )}
                  </td>
                </tr>

                <tr>
                  <td className="cell-metric">Prediction Confidence (%)</td>
                  <td className="cell-value">
                    {rfLive.latest_confidence != null
                      ? `${(rfLive.latest_confidence * 100).toFixed(1)}%`
                      : "N/A"}
                  </td>
                  <td className="cell-value">
                    {xgbLive.latest_confidence != null
                      ? `${(xgbLive.latest_confidence * 100).toFixed(1)}%`
                      : "N/A"}
                  </td>
                </tr>

                <tr>
                  <td className="cell-metric">Inference Latency (ms)</td>
                  <td className="cell-value">{formatLatency(rfLive.inference_latency_ms)}</td>
                  <td className="cell-value">{formatLatency(xgbLive.inference_latency_ms)}</td>
                </tr>

                <tr>
                  <td className="cell-metric">Total Predictions</td>
                  <td className="cell-value">{formatCount(rfLive.total_predictions)}</td>
                  <td className="cell-value">{formatCount(xgbLive.total_predictions)}</td>
                </tr>

                <tr>
                  <td className="cell-metric">Normal Predictions</td>
                  <td className="cell-value">{formatCount(rfLive.normal_predictions)}</td>
                  <td className="cell-value">{formatCount(xgbLive.normal_predictions)}</td>
                </tr>

                <tr>
                  <td className="cell-metric">Anomaly Predictions</td>
                  <td className="cell-value">{formatCount(rfLive.anomaly_predictions)}</td>
                  <td className="cell-value">{formatCount(xgbLive.anomaly_predictions)}</td>
                </tr>

                <tr>
                  <td className="cell-metric">Prediction Agreement (%)</td>
                  <td className="cell-value" colSpan={2}>
                    <div className="shared-agreement-wrap">
                      <span className="shared-agreement-val">
                        {liveComp.agreement_percentage != null
                          ? `${liveComp.agreement_percentage.toFixed(1)}%`
                          : "N/A"}
                      </span>
                      {liveComp.total_compared > 0 && (
                        <span className="shared-agreement-detail">
                          ({liveComp.agreement_count} / {liveComp.total_compared} matched)
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
