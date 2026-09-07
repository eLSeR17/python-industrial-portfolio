# Data Directory

This directory is a placeholder for supply chain data files used in development and testing.

## Expected Data Formats

### CSV: Node Definitions
```csv
id,name,type,latitude,longitude,capacity,fixed_cost,variable_cost,demand
SUP-01,Steel Corp Midwest,supplier,41.88,-87.63,50000,15000,2.50,
WH-01,Chicago Hub,warehouse,41.88,-87.63,30000,25000,0.80,
CUS-01,Retail North,customer,45.50,-73.57,,,2000
```

### CSV: Edge/Route Definitions
```csv
source,target,transport_mode,distance_km,cost_per_unit,transit_time_hours,capacity,reliability
SUP-01,WH-01,truck,0.5,0.10,0.5,50000,1.0
WH-01,CUS-01,truck,1200,3.00,12,5000,0.95
```

### JSON: Historical Demand
```json
{
  "node_id": "CUS-01",
  "period": "2024-01",
  "demand": 1850
}
```

## Data Sources

Replace with your organization's data:
- **ERP export**: node and edge definitions
- **Historical shipments**: actual transport costs and times
- **Demand history**: 12-24 months of time series data
- **Supplier scorecards**: quality, delivery, pricing records
