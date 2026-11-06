import { Server } from "lucide-react";
import "./NetworkTopology.css";

export function NetworkTopology({ devices, onOpenDevice, compact = false, dashboard = false, gatewayConnected = false }) {
  const shown = devices.slice(0, compact ? 5 : 8)
  const statusColor = (status) => status === 'online' ? '#47bd91' : status === 'warning' || status === 'stale' ? '#e6b45e' : status === 'critical' ? '#e66f69' : '#77838e'
  const large = !compact
  const nodeWidth = large ? 214 : 102
  const nodeHeight = large ? 72 : 62
  const pillHeight = large ? 36 : 19
  const rowHeight = nodeHeight + 21 + pillHeight
  const rowGap = 24
  const columns = !dashboard && shown.length > 3 ? 2 : 1
  const rows = Math.ceil(Math.max(shown.length, 1) / columns)
  const stackHeight = rows * rowHeight + (rows - 1) * rowGap
  const width = dashboard ? 560 : 960
  const height = large ? Math.max(dashboard ? 260 : 300, stackHeight + 72) : 290
  const stackTop = (height - stackHeight) / 2
  const firstCenter = stackTop + nodeHeight / 2
  const lastCenter = stackTop + (rows - 1) * (rowHeight + rowGap) + nodeHeight / 2
  const deviceX = dashboard ? 330 : columns === 2 ? 480 : 590
  const positions = large
    ? shown.map((device, index) => ({
      device,
      x: deviceX + (index % columns) * (nodeWidth + 48),
      y: stackTop + Math.floor(index / columns) * (rowHeight + rowGap),
    }))
    : shown.map((device, index) => ({ device, x: 286 + index * 126, y: index % 2 ? 215 : 102 }))
  const gateway = large
    ? { x: dashboard ? 20 : 170, y: (firstCenter + lastCenter) / 2 - 38 }
    : { x: 65, y: 157 }
  const connection = ({ x, y }) => {
    const endX = x
    const endY = y + nodeHeight / 2
    const startX = gateway.x + (compact ? 84 : 116)
    const startY = gateway.y + (compact ? 27 : 38)
    const middleX = (startX + endX) / 2
    return `M ${startX} ${startY} C ${middleX} ${startY}, ${middleX} ${endY}, ${endX} ${endY}`
  }
  return (
    <div className={`topology-canvas ${compact ? 'compact-canvas' : 'topology-large'} ${dashboard ? 'dashboard-canvas' : ''}`}>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" className="topology-svg" role="img" aria-label="Interactive network topology diagram">
        <defs><pattern id="topologyGrid" width="22" height="22" patternUnits="userSpaceOnUse"><path d="M 22 0 L 0 0 0 22" fill="none" stroke="rgba(125,155,175,.055)" strokeWidth="1" /></pattern></defs>
        <rect width={width} height={height} fill="url(#topologyGrid)" />
        {positions.map(({ device, x, y }) => {
          const connectionStatus = device.online ? "online" : device.status;
          return <g key={`link-${device.id}`}><path d={connection({ x, y })} className={`topology-link ${connectionStatus === 'online' ? 'healthy-link' : `${connectionStatus}-link`}`} /><circle r="3.3" fill={statusColor(connectionStatus)} className="link-pulse"><animateMotion dur={`${4 + Number(device.id.slice(-1)) % 3}s`} repeatCount="indefinite" path={connection({ x, y })} /></circle></g>;
        })}
        <g className="gateway-node" transform={`translate(${gateway.x},${gateway.y})`}>
          <rect width={compact ? 84 : 116} height={compact ? 54 : 76} rx="6" /><circle cx={compact ? 19 : 25} cy={compact ? 27 : 38} r="13" />
          <g transform={`translate(${compact ? 11 : 17},${compact ? 19 : 30})`}><Server size={16} color="#69bfd4" /></g>
          <text x={compact ? 39 : 48} y={compact ? 24 : 33}>MQTT</text><text x={compact ? 39 : 48} y={compact ? 38 : 49} className="topology-sub">GATEWAY</text>
          <circle cx={compact ? 72 : 101} cy="10" r="4" fill={gatewayConnected ? "#47bd91" : "#75828c"} className={gatewayConnected ? "gateway-light" : ""} />
        </g>
        {positions.map(({ device, x, y }) => {
          const connectionStatus = device.online ? "online" : device.status;
          const failedSensor = device.sensors.find((sensor) => ['critical', 'offline'].includes(sensor.status))
          const sensorStatus = failedSensor?.status ?? 'healthy'
          return (
            <g className={`device-node ${connectionStatus}`} key={device.id} transform={`translate(${x},${y})`} onClick={() => onOpenDevice(device.id)} tabIndex="0" role="button" aria-label={`Open ${device.id} ${device.name}`} onKeyDown={(event) => event.key === 'Enter' && onOpenDevice(device.id)}>
              <rect className="device-node-bg" width={nodeWidth} height={nodeHeight} rx="6" />
              <rect className="node-accent" width="3" height={nodeHeight} rx="2" fill={statusColor(connectionStatus)} />
              <circle cx="20" cy={nodeHeight / 2} r="10" fill={statusColor(connectionStatus)} opacity=".15" /><circle cx="20" cy={nodeHeight / 2} r="4" fill={statusColor(connectionStatus)} />
              <text x="38" y={nodeHeight / 2 - 5} className="node-id">{device.id}</text>
              <text x="38" y={nodeHeight / 2 + 10} className="node-name">{device.name.length > 14 ? `${device.name.slice(0, 13)}…` : device.name}</text>
              <text x={nodeWidth - 8} y="15" textAnchor="end" className="node-health">{connectionStatus === 'offline' || connectionStatus === 'stale' ? 'OFFLINE' : device.online ? 'ONLINE' : Number.isFinite(device.health) ? `${device.health}%` : '—'}</text>
              {failedSensor && <g className={`sensor-alert-flag ${sensorStatus}`} transform={large ? `translate(${nodeWidth - 88},-9)` : `translate(${nodeWidth - 48},-7)`}><rect width={large ? 92 : 52} height={large ? 18 : 14} rx="3" /><text x={large ? 46 : 26} y={large ? 12.5 : 10} textAnchor="middle">SENSOR FAULT</text></g>}
              {!compact && <g className="child-nodes" transform={`translate(0,${nodeHeight + 8})`}>
                <path d={`M ${nodeWidth / 2} 0 V 8 M ${nodeWidth * .24} 8 H ${nodeWidth * .76} M ${nodeWidth * .24} 8 V 13 M ${nodeWidth * .76} 8 V 13`} />
                <g className={`asset-node ${device.motorHealth === null ? 'offline' : device.motorHealth < 60 ? 'critical' : device.motorHealth < 80 ? 'warning' : 'healthy'}`} transform={`translate(3,13)`}>
                  <rect width={nodeWidth / 2 - 5} height={pillHeight} rx={large ? 5 : 3} /><text x={large ? 10 : 4} y={large ? 14 : 8}>ASSET</text><text x={large ? 10 : 4} y={large ? 28 : 15}>{device.asset.toUpperCase().slice(0, 11)}</text>
                </g>
                <g className={`sensor-node ${sensorStatus}`} transform={`translate(${nodeWidth / 2 + 2},13)`}>
                  <rect width={nodeWidth / 2 - 5} height={pillHeight} rx={large ? 5 : 3} /><text x={large ? 10 : 4} y={large ? 14 : 8}>SENSORS · {device.sensorsCount}</text><text x={large ? 10 : 4} y={large ? 28 : 15}>{device.status === 'stale' || connectionStatus === 'offline' ? 'OFFLINE' : failedSensor ? failedSensor.kind.toUpperCase() + ' FAULT' : 'HEALTHY'}</text>
                </g>
              </g>}
            </g>
          )
        })}
      </svg>
      <div className="canvas-hint"><span>Click a device to inspect</span><span><span className="status-dot healthy" /> {shown.length} {shown.length === 1 ? 'device' : 'devices'} visible</span></div>
    </div>
  )
}
