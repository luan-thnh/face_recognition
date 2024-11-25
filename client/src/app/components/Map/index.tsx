import { Icon, LatLngExpression } from 'leaflet';
import React, { useEffect } from 'react';
import { MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

interface MapProps {
  latitude: number;
  longitude: number;
}

const customIcon = new Icon({
  iconUrl: '/location.png',
  iconSize: [24, 24],
});

const MapUpdater: React.FC<MapProps> = ({ latitude, longitude }) => {
  const map = useMap();

  useEffect(() => {
    if (latitude && longitude) {
      const position: LatLngExpression = [latitude, longitude];
      map.flyTo(position, 13);
    }
  }, [latitude, longitude, map]);

  return null;
};

const Map: React.FC<MapProps> = ({ latitude, longitude }) => {
  const position: LatLngExpression = [latitude, longitude];

  return (
    <div className="ring-2 ring-blue-600/60 ring-offset-2 rounded-lg">
      <MapContainer
        center={position}
        zoom={13}
        scrollWheelZoom={false}
        className="w-[480px] h-[360px] border rounded-lg shadow-lg"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Marker position={position} icon={customIcon}>
          <Popup>
            Your location: {latitude.toFixed(6)}, {longitude.toFixed(6)}
          </Popup>
        </Marker>
        <MapUpdater latitude={latitude} longitude={longitude} />
      </MapContainer>
    </div>
  );
};

export default Map;
