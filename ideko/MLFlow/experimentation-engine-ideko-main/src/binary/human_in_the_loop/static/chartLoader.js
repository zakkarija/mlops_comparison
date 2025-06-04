//API_URL = "http://localhost:5000"
API_URL = "http://" + backend_ip_and_port

async function fetchData() {
    try {
        const response = await fetch(API_URL + "/get_csv_data");
        const data = await response.json();
        const time = data.time.map(Number);  
        
        Highcharts.chart("container", {
            title: { text: null },
            credits: { enabled: false },
            chart: { zoomType: "xy" },
            xAxis: {
                labels: { enabled: false }, 
            },
            yAxis: [
                { 
                    title: { text: "Position" },
                    min: Math.min(...data.f1, ...data.f2, ...data.f4), 
                    max: Math.max(...data.f1, ...data.f2, ...data.f4), 
                    startOnTick: false,
                    endOnTick: false
                },
                { 
                    title: { text: "Intensity" },
                    opposite: true,
                    min: Math.min(...data.f3), 
                    max: Math.max(...data.f3), 
                    startOnTick: false,
                    endOnTick: false,
                    opposite: true
                }
            ],
            series: [
                { name: "Encoder Position", data: data.f1.map(Number), yAxis: 0, color: "red" },
                { name: "Ruler Position", data: data.f2.map(Number), yAxis: 0, color: "green" },
                { name: "Commanded Position", data: data.f4.map(Number), yAxis: 0, color: "orange", lineWidth: 2 },
                { name: "Intensity", data: data.f3.map(Number), yAxis: 1, color: "blue" }
            ]
        });
    } catch (error) {
        console.error("Error fetching file:", error);
    }
}
fetchData();