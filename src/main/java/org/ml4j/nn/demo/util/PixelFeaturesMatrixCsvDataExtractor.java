/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.nn.demo.util;

import org.ml4j.util.NumericFeaturesMatrixCsvDataExtractor;

/**
 * Extracts the Mnist digit image data from a row in the CSV file.
 * 
 * @author Michael Lavelle
 */
public class PixelFeaturesMatrixCsvDataExtractor extends NumericFeaturesMatrixCsvDataExtractor {

  @Override
  public double[] createData(String[] csvAttributes) {
    double[] rawData = super.createData(csvAttributes);
    // Reverse zeros and ones in the csv file so that pixels of pen strokes map to active neurons
    double[] pixelActivationData = new double[rawData.length];
    for (int i = 0; i < rawData.length; i++) {
      pixelActivationData[i] = rawData[i] == 0 ? 1 : 0;
    }
    return pixelActivationData;
  }
}
