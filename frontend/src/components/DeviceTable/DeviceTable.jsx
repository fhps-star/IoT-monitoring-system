import { ChevronRight, Gauge } from "lucide-react";
import { Badge } from "../StatusBadge/StatusBadge.jsx";
import { useState } from "react";
import { statusLabel } from "../../utils/status.js";
import "./DeviceTable.css";

export function DeviceTable({ devices, onOpenDevice, compact = false }) {
  const [sortKey, setSortKey] = useState('id')
  const [sortDirection, setSortDirection] = useState('asc')
  if (!devices.length) return <div className="empty-state"><div><Gauge size={18} /></div><strong>No devices found</strong><span>Adjust the search or selected status filter.</span></div>
  const columns = [
    ['DEVICE', 'id'], ['ASSET', 'asset'], ['STATUS', 'status'], ['SENSORS', 'sensorsCount'],
    ['IP ADDRESS', 'ip'], ['PROTOCOL', 'protocol'], ['LATENCY', 'latency'], ['PACKET LOSS', 'packetLoss'], ['HEALTH', 'deviceHealth'],
  ]
  const sortedDevices = [...devices].sort((left, right) => {
    const first = left[sortKey]
    const second = right[sortKey]
    const comparison = typeof first === 'number' ? first - second : String(first).localeCompare(String(second))
    return sortDirection === 'asc' ? comparison : -comparison
  })
  const sortBy = (key) => {
    if (sortKey === key) setSortDirection((direction) => direction === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDirection('asc') }
  }
  return (
    <div className="table-scroll">
      <table className={`data-table device-table ${compact ? 'compact-table' : ''}`}>
        <thead><tr>{columns.map(([label, key]) => <th key={key} aria-sort={sortKey === key ? sortDirection === 'asc' ? 'ascending' : 'descending' : 'none'}><button className="sortable-heading" onClick={() => sortBy(key)}>{label}{sortKey === key && <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>}</button></th>)}<th /></tr></thead>
        <tbody>{sortedDevices.map((device) => {
          return (
            <tr key={device.id} onClick={() => onOpenDevice(device.id)} tabIndex="0" onKeyDown={(event) => event.key === 'Enter' && onOpenDevice(device.id)}>
              <td><div className="device-id-cell"><span><strong>{device.id}</strong><small>{device.name}</small></span></div></td>
              <td>{device.asset}<small>Production line {device.line}</small></td>
              <td><Badge value={statusLabel(device.status)} /></td>
              <td>{device.sensorsCount} <span className="table-muted">configured</span></td>
              <td className="mono-cell">{device.ip}</td><td>{device.protocol}</td>
              <td className={device.latency > 80 ? 'value-warning' : ''}>{Number.isFinite(device.latency) ? `${device.latency} ms` : '—'}</td>
              <td className={device.packetLoss > 1 ? 'value-warning' : device.status === 'offline' ? 'value-danger' : ''}>{Number.isFinite(device.packetLoss) ? `${device.packetLoss.toFixed(2)}%` : '—'}</td>
              <td>{device.deviceHealth
                ? <Badge value={device.deviceHealth} />
                : Number.isFinite(device.health)
                  ? <div className="health-cell"><span className="health-track"><i className={device.status} style={{ width: `${device.health}%` }} /></span><strong>{device.health}%</strong></div>
                  : "—"}</td>
              <td><ChevronRight size={15} className="row-chevron" /></td>
            </tr>
          )
        })}</tbody>
      </table>
    </div>
  )
}
