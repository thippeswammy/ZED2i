import os
import pyzed.sl as sl

all_file = {}


def openKMLFile(file_path):
    """
    Open and write KML header in file
    """
    if file_path in all_file:
        return
    file_object = open(file_path, 'w')
    file_header = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"

    file_header += "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n"
    file_header += "\t<Document>\n"
    file_header += "\t\t<name>generated_path</name>\n"
    file_header += "\t\t<description>Paths generated by ZED SDK</description>\n"
    file_header += "<Style id=\"my-awesome-light-blue\">\n"
    file_header += "\t<LineStyle>\n"
    file_header += "\t\t<color>57F2DA</color>\n"
    file_header += "\t\t<width>4</width>\n"
    file_header += "\t</LineStyle>\n"
    file_header += "\t<PolyStyle>\n"
    file_header += "\t\t<color>57F2DA</color>\n"
    file_header += "\t</PolyStyle>\n"
    file_header += "</Style>\n"
    file_header += "<Placemark>\n"
    file_header += "\t<name>generated_path</name>\n"
    file_header += "\t<description>generated by zed SDK</description>\n"
    file_header += "\t<styleUrl>#57F2DA</styleUrl>\n"
    file_header += "\t<LineString>\n"
    file_header += "\t\t<extrude>1</extrude>\n"
    file_header += "\t\t<tessellate>1</tessellate>\n"
    file_header += "\t\t<altitudeMode>absolute</altitudeMode>\n"
    file_header += "\t<coordinates> "
    file_object.write(file_header)
    return file_object


def closeAllKMLFiles():
    """
    Close all KML file writer and place KML files footer
    """
    for file_name, file_object in all_file:
        file_footer = "\t</coordinates>\n"
        file_footer += "\t</LineString>\n"
        file_footer += "</Placemark>\n"
        file_footer += "\t</Document>\n"
        file_footer += "</kml>\n"
        file_object.write(file_footer)
        file_object.close()


def saveKMLData(file_path, gnss_data):
    if not file_path in all_file:
        all_file[file_path] = openKMLFile(file_path)
    assert gnss_data["longitude"] is not None
    assert gnss_data["latitude"] is not None
    assert gnss_data["altitude"] is not None
    latlng_content = str(gnss_data["longitude"]) + ", " + str(
        gnss_data["latitude"]) + ", " + str(gnss_data["altitude"]) + "\n"
    all_file[file_path].write(latlng_content)
