import React from 'react';
import DisasterPredictor from '../components/DisasterPredictor';
import { Activity, Target, AlertTriangle, RadioTower, Maximize, Orbit, ArrowDown, MapPin, Navigation } from 'lucide-react';

export default function Earthquake() {
    return (
        <DisasterPredictor
            title="Seismic Intelligence"
            description="Predict high-magnitude earthquakes (≥7.0) and severe destruction utilizing full NASA/USGS telemetry."
            disasterType="earthquake"
            themeColor="amber"
            icon={ArrowDown}
            featuresMeta={{
                magnitude: { icon: Activity,     label: "Magnitude",        desc: "Richter magnitude estimate",         min: 0,   max: 10,  step: 0.1, default: 6.8  },
                cdi:       { icon: Target,       label: "Max CDI",          desc: "Community Internet Intensity Map",   min: 0,   max: 10,  step: 1,   default: 5    },
                mmi:       { icon: AlertTriangle,label: "Max MMI",          desc: "Modified Mercalli Intensity",        min: 0,   max: 10,  step: 1,   default: 6    },
                sig:       { icon: RadioTower,   label: "Significance",     desc: "Event significance score",           min: 0,   max: 3000,step: 10,  default: 800  },
                nst:       { icon: Orbit,        label: "Seismic Stations", desc: "Number of reporting stations",       min: 0,   max: 500, step: 1,   default: 40   },
                dmin:      { icon: Maximize,     label: "Min Distance",     desc: "Distance to nearest station (deg)",  min: 0,   max: 50,  step: 0.1, default: 2.0  },
                gap:       { icon: Target,       label: "Azimuthal Gap",    desc: "Largest gap between stations (deg)", min: 0,   max: 360, step: 1,   default: 30   },
                depth:     { icon: ArrowDown,    label: "Focal Depth (km)", desc: "Hypocenter depth below surface",     min: 0,   max: 700, step: 1,   default: 20   },
                latitude:  { icon: MapPin,       label: "Latitude",         desc: "N/S Geolocation (-90 to 90)",        min: -90, max: 90,  step: 0.1, default: 35.0  },
                longitude: { icon: Navigation,   label: "Longitude",        desc: "E/W Geolocation (-180 to 180)",      min: -180,max: 180, step: 0.1, default: -118.0}
            }}
        />
    );
}
