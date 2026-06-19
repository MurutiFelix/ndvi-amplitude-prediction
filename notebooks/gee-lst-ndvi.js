// Define start date and number of months (20 years * 12 months = 240)
var startDate = ee.Date('2006-01-01');
var totalMonths = 60; // 10-year chunk -- next starts at 2011

// Load collections
var ndviCol = ee.ImageCollection("MODIS/061/MOD13A3").select('NDVI');

// FIX: Swapped out non-existent MOD11A3 for the official 1km MOD11A2 8-Day collection
var lstCol = ee.ImageCollection("MODIS/061/MOD11A2").select('LST_Day_1km');

// Loop through all specified months
for (var i = 0; i < totalMonths; i++) {
  
  // Calculate current month date range
  var currentMonthStart = startDate.advance(i, 'month');
  var currentMonthEnd = currentMonthStart.advance(1, 'month');
  
  // Create a date string format for filenames (e.g., 2006_01)
  var dateString = currentMonthStart.format('yyyy_MM').getInfo();
  
  // Filter, scale, and clip NDVI for this specific month
  var monthlyNDVI = ndviCol.filterDate(currentMonthStart, currentMonthEnd)
    .first() // Works perfectly because MOD13A3 has 1 image per month
    .multiply(0.0001)
    .clip(aoi);
    
  // FIX: Filter, calculate monthly MEAN of the 8-day snapshots, and clip
  var monthlyLST = lstCol.filterDate(currentMonthStart, currentMonthEnd)
    .mean() // Changed from .first() to .mean() to average the 8-day images into a monthly mean
    .multiply(0.02).subtract(273.15)
    .clip(aoi);

  // Export individual NDVI image
  Export.image.toDrive({
    image: monthlyNDVI,
    description: 'NDVI_' + dateString,
    folder: 'Dynamics',
    scale: 1000,
    region: aoi.geometry(),
    maxPixels: 1e13
  });

  // Export individual LST image (Will now execute perfectly)
  Export.image.toDrive({
    image: monthlyLST,
    description: 'LST_' + dateString,
    folder: 'Dynamics',
    scale: 1000,
    region: aoi.geometry(),
    maxPixels: 1e13
  });
}