/*
 * Copyright 2015 the original author or authors.
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

import org.ml4j.util.MultiClassLabelsMatrixCsvDataExtractor;

/**
 * Maps text files containing one single digit (range 0-9) on each row to multiclass label vectors.
 * 
 * @author Michael Lavelle
 *
 */
public class SingleDigitLabelsMatrixCsvDataExtractor
    extends MultiClassLabelsMatrixCsvDataExtractor {

  public SingleDigitLabelsMatrixCsvDataExtractor() {
    super(10);
  }

  @Override
  protected int getLabelIndex(String[] csvAttributes) {
    return Integer.parseInt(csvAttributes[0]);
  }
}
