import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { catchError } from 'rxjs/operators';

import { Question } from './_models/quesiton';  
import { questionGenerationRequest } from './_models/questionGenerationRequest';

@Injectable({
  providedIn: 'root'
})
export class QuestionGenerationService {

  private questionGenerationUrl = 'http://localhost:9002/generate';  // URL to web api
  // private questionGenerationUrl = 'api/heroes';  // URL to web api
  httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
  };

  constructor( private http: HttpClient ) { }

  generate(req: questionGenerationRequest): Observable<Question[]> {
    return this.http.post<Question[]>(this.questionGenerationUrl, req, this.httpOptions).pipe(
      catchError((error: HttpErrorResponse) => {
        console.error('❌ Lỗi kết nối tới Backend:', error.message);
        console.error('👉 Đảm bảo Flask đang chạy tại http://localhost:9002');
        alert('Lỗi kết nối Backend! Đảm bảo python api_gateway.py đang chạy. Xem Console (F12) để biết chi tiết.');
        return throwError(error);
      })
    );
  }

}
