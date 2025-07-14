import xml.etree.ElementTree as ET # XML parsing library
import os

def split_reviews(input_file, output_dir="output_parts", parts=10):
    # Parse the original XML
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Get all reviews
    reviews = root.findall("Review")
    total_reviews = len(reviews)
    reviews_per_part = total_reviews // parts


    os.makedirs(output_dir, exist_ok=True)

    for i in range(parts):
        start_idx = i * reviews_per_part
        end_idx = start_idx + reviews_per_part

        part_root = ET.Element("Reviews")
        for review in reviews[start_idx:end_idx]:
            part_root.append(review)

        part_tree = ET.ElementTree(part_root)
        part_file = os.path.join(output_dir, f"part{i + 1}.xml")
        part_tree.write(part_file, encoding="utf-8", xml_declaration=True)

    print(f"Split completed. {parts} parts written to '{output_dir}'.")

if __name__ == "__main__":
    input_file_path = "./ABSA16_Restaurants_Train_SB1_v2.xml"
    split_reviews(input_file_path)
